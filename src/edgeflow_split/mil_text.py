from __future__ import annotations

MIL_HEADER = (
    'program(1.3)\n'
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
    '{\n'
)

MIL_BODY_INDENT = "        "

MIL_CONV_CONSTS = (
    '        string pt = const()[name=string("pt"), val=string("valid")];\n'
    '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '        int32 gr = const()[name=string("gr"), val=int32(1)];\n'
)


def mil_blobfile(path: str) -> str:
    return f'BLOBFILE(path=string("{path}"), offset=uint64(64))'


def mil_main_func_open(input_decls: str) -> str:
    return f"    func main<ios18>({input_decls}) {{\n"
