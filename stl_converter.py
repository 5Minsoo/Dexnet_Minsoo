import cadquery as cq

# STEP 파일 로드
result = cq.importers.importStep("/home/minsoo/bin_picking1/ROBOTIQ_HAND-E_DEFEATURE_20181026-Sep-06-2024-02-31-01-9043-PM.step")

# STL로 내보내기
cq.exporters.export(result, "output.stl")

print("변환 완료: output.stl")