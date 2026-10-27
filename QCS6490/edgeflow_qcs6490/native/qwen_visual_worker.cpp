// Visual-only Qwen2.5-VL-3B CPU worker. No Llm/Omni/llama loading or HTP path.
// CLI: qwen_visual_worker /absolute/path/visual.mnn (adjacent .mnn.weight).
// stdin: ENCODE<TAB>patches<TAB>positions<TAB>mask<TAB>window_index<TAB>output
// stdout: READY mnn_cpu 2 225 2048, then RES <onForward/readMap milliseconds>.
// QUIT or stdin EOF stops the persistent worker; failures print FATAL and exit.

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Module.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

template <typename T>
static VARP read_input(const std::string& path, const std::vector<int>& shape) {
    size_t count = 1;
    for (int dim : shape) count *= static_cast<size_t>(dim);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(T))) {
        throw std::runtime_error("wrong input byte size: " + path);
    }
    auto input = _Input(shape, NCHW, halide_type_of<T>());
    T* values = input.get() ? input->template writeMap<T>() : nullptr;
    if (!values) throw std::runtime_error("cannot map visual input");
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(values), count * sizeof(T))) {
        throw std::runtime_error("cannot read visual input: " + path);
    }
    return input;
}

static void write_output(const std::string& path, const float* values) {
    constexpr size_t count = 225 * 2048;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) throw std::runtime_error("non-finite visual output");
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(reinterpret_cast<const char*>(values), count * sizeof(float))) {
        throw std::runtime_error("cannot write visual output");
    }
    file.close();
    if (!file) throw std::runtime_error("cannot close visual output");
}

int main(int argc, char** argv) {
    if ((argc != 2 && argc != 3) || std::string(argv[1]) == "--help") {
        std::cerr << "usage: qwen_visual_worker /path/to/visual.mnn [cpu|opencl]\n";
        return argc == 2 ? 0 : 2;
    }
    try {
        const std::string model = argv[1];
        const bool opencl = argc == 3 && std::string(argv[2]) == "opencl";
        if (argc == 3 && !opencl && std::string(argv[2]) != "cpu") throw std::runtime_error("backend must be cpu or opencl");
        const auto type = opencl ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU;
        const int threads = opencl ? 65 : 2; // Buffer mode, frozen tuning; not 65 host threads.
        BackendConfig backend;
        backend.precision = opencl ? BackendConfig::Precision_Low : BackendConfig::Precision_High;
        auto executor = Executor::newExecutor(type, backend, threads);
        ExecutorScope scope(executor);
        ScheduleConfig schedule;
        schedule.type = type;
        schedule.numThread = threads;
        schedule.backendConfig = &backend;
        std::shared_ptr<Executor::RuntimeManager> runtime(
            Executor::RuntimeManager::createRuntimeManager(schedule));
        if (!runtime) throw std::runtime_error("cannot create CPU runtime manager");
        runtime->setExternalFile(model + ".weight");
        Module::Config config;
        config.shapeMutable = true;
        config.rearrange = !opencl; // Match Omni's placement-dependent load mode.
        std::unique_ptr<Module> visual(Module::load({}, {}, model.c_str(), runtime, &config));
        if (!visual) throw std::runtime_error("cannot load trained visual.mnn");
        visual->setIsTraining(false);
        const auto* info = visual->getInfo();
        if (!info || info->inputNames != std::vector<std::string>{
                "patches", "position_ids", "attention_mask", "window_index"} ||
                info->outputNames != std::vector<std::string>{"image_embeds"}) {
            throw std::runtime_error("requires four-input Qwen2.5-VL visual module");
        }
        std::cout << "READY mnn_" << (opencl ? "opencl" : "cpu") << " " << threads << " 225 2048\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            if (line.size() > 4096) throw std::runtime_error("visual request exceeds bound");
            std::istringstream request(line);
            std::vector<std::string> fields;
            std::string field;
            while (std::getline(request, field, '\t')) fields.push_back(field);
            if (fields.size() != 6 || fields[0] != "ENCODE") {
                throw std::runtime_error("expected ENCODE and five tab-separated paths");
            }
            std::vector<VARP> inputs = {
                read_input<float>(fields[1], {900, 1176}),
                read_input<int32_t>(fields[2], {2, 900}),
                read_input<float>(fields[3], {2, 1, 900, 900}),
                read_input<int32_t>(fields[4], {225})
            };
            const auto start = std::chrono::steady_clock::now();
            const auto outputs = visual->onForward(inputs);
            if (outputs.size() != 1 || !outputs[0].get()) {
                throw std::runtime_error("visual module did not return image_embeds");
            }
            const auto* output_info = outputs[0]->getInfo();
            if (!output_info || output_info->dim != std::vector<int>{225, 1, 2048} ||
                    output_info->size != 225 * 2048 || output_info->type.code != halide_type_float ||
                    output_info->type.bits != 32) {
                throw std::runtime_error("image_embeds must be float32 [225,1,2048]");
            }
            const float* values = outputs[0]->readMap<float>();
            if (!values) throw std::runtime_error("cannot map image_embeds");
            const double vision_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            // The exported graph already includes merger + argsort(window_index).
            // Keep these 225 rows unchanged; Python adds only real word markers.
            write_output(fields[5], values);
            std::cout << "RES " << std::setprecision(17) << vision_ms << '\n' << std::flush;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen_visual_worker: " << error.what() << '\n';
        std::cout << "FATAL " << error.what() << '\n' << std::flush;
        return 1;
    }
}
