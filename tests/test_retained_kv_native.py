"""Compile the actual retained export/import branch against a metadata stub.

The independent HandoffImporter test executes the real per-layer staging path;
this test exercises worker ownership/position checks and the real snapshot IO.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_metadata_tool(tmp_path_factory):
    native = Path(__file__).resolve().parents[1] / "RK3588/edgeflow_rk3588/runtime/tabm/native"
    common = (native / "mnn_handoff_state.hpp").read_text()
    structs = common[common.index("struct HandoffLayer"):common.index("static size_t bounded_kv_count")]
    worker = (native / "tabm_decoder_worker.cpp").read_text()
    b64 = worker[worker.index("static std::string b64_decode"):worker.index("struct ReqResult")]
    retained = (native / "retained_context.hpp").read_text()
    start = retained.index('    if (op == "export" || op == "import" || op == "cache_import")')
    branch = retained[start:retained.index("\n#endif", start)]
    directory = tmp_path_factory.mktemp("retained-kv-native")
    source = directory / "test.cpp"
    source.write_text(
        '#include <string>\n#include <map>\n#include <vector>\n#include <memory>\n'
        '#include <stdexcept>\n#include <cstring>\n#include <cassert>\n#include <cstdint>\n'
        '#include "mnn_kv_transfer.hpp"\n'
        'static void require(bool c,const std::string& m){if(!c)throw std::runtime_error(m);}\n'
        + structs + b64 + r'''
#include "handoff_snapshot_file.hpp"
struct Llm { size_t positions=0; size_t getCurrentHistory(){return positions;} };
struct Importer { void begin(Llm& llm,const HandoffSnapshot& state){llm.positions=state.positions;} };
struct Transfer {
    bool donor=false; std::string model_key="model_a";
    struct { HandoffSnapshot snapshot; } capture;
    WorkerKVSnapshot imported;
    std::unique_ptr<Importer> importer{new Importer};
    void clear(){imported=WorkerKVSnapshot();}
};
struct Root {
    bool valid=true,active=true; size_t end=4,physical=4;
    int hidden=896; int64_t version=2; std::string stream="stream_a";
    std::map<long long,size_t> version_ends{{1,3},{2,4}};
    std::vector<int> prefix{10,11}; std::unique_ptr<Transfer> transfer{new Transfer};
};
struct Result { bool ok=false; std::string err; size_t exported_bytes=0; };
static void execute(Llm& llm,Root& root,Result& r,const std::string& op,
                    const std::string& path,int64_t version,int end=0) {
    std::vector<std::string> fields{"CTX","1",op,"stream_a",std::to_string(version),b64_encode(path),std::string(64,'a')};
    auto& reply=r;
    if(end){fields.push_back("1");fields.push_back(std::to_string(end));}
    if (op=="import" && version==9) {fields[4]="2";fields[6]=std::string(64,'b');version=2;}
    if (op=="cache_import") {
        fields[3]="stream_b";fields[6]=std::string(64,'b');
        fields.push_back("stream_a");fields.push_back("2");fields.push_back(std::string(64,'a'));
        if(version==8){fields[4]="2";fields[7]="wrong_source";version=2;}
    }
''' + branch + r'''
}
int main(int argc,char** argv) {
    const std::string mode=argv[1],path=argv[2];
    Llm llm; Root donor; donor.transfer->donor=true;
    donor.transfer->capture.snapshot.positions=4;
    for(int i=0;i<2;++i){
        auto& layer=donor.transfer->capture.snapshot.layers["attention_"+std::to_string(i)];
        layer.tokens=4;layer.heads=1;layer.dimension=2;
        layer.key={1,2,3,4,5,6,7,8};layer.value=layer.key;
    }
    Result exported;execute(llm,donor,exported,"export",path,2,mode=="historical"?3:mode=="partial_frame"?2:0);
    if(mode=="partial_frame"){assert(!exported.ok && donor.end==4);return 0;}
    assert(exported.ok && donor.end==4 && donor.version==2);
    if(mode=="historical"){
        assert(exported.exported_bytes==96);
        const auto file=read_worker_kv_snapshot(path);
        assert(file.version==1 && file.state.positions==3);
        assert(file.state.layers.begin()->second.key==std::vector<float>({1,2,3,4,5,6}));
        assert(donor.transfer->capture.snapshot.layers.begin()->second.key.size()==8);
        return 0;
    }
    Root receiver; receiver.end=receiver.physical=0; receiver.version=0;
    if(mode=="wrong_model")receiver.transfer->model_key="other_model";
    if(mode=="wrong_prefix")receiver.prefix={99};
    if(mode=="wrong_backend")receiver.transfer->donor=true;
    Result imported;
    const bool cached = mode=="cached_alias" || mode=="wrong_cached_source";
    execute(llm,receiver,imported,cached?"cache_import":"import",path,
            mode=="wrong_version"?3:(mode=="wrong_content"?9:(mode=="wrong_cached_source"?8:2)));
    if(mode=="roundtrip" || mode=="cached_alias"){
        assert(imported.ok && receiver.end==4 && receiver.physical==4);
        assert(receiver.version==2 && receiver.stream==(cached?"stream_b":"stream_a") && receiver.hidden==896);
        assert(receiver.transfer->imported.state.layers.size()==2 && llm.positions==4);
    } else {
        assert(!imported.ok && !imported.err.empty() && receiver.end==0 && llm.positions==0);
    }
}
'''
    )
    binary = directory / "metadata"
    subprocess.run([shutil.which("c++"), "-std=c++17", "-I", str(native), str(source),
                    "-o", str(binary)], capture_output=True, text=True, check=True)
    return binary


@pytest.mark.parametrize("mode", ["roundtrip", "wrong_model", "wrong_prefix", "wrong_backend", "wrong_version", "wrong_content", "cached_alias", "wrong_cached_source", "historical", "partial_frame"])
def test_native_binding_precedes_state_mutation(native_metadata_tool, tmp_path, mode):
    subprocess.run([native_metadata_tool, mode, tmp_path / "state.kv"], check=True)
