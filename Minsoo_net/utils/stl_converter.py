import cadquery as cq
import os

# 1. 설정: STEP 파일들이 들어있는 폴더 경로
input_folder = "/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object"
output_folder = input_folder

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. 폴더 내 모든 파일 탐색
for filename in os.listdir(input_folder):
    # 확장자가 .step 또는 .stp인 파일만 필터링 (대소문자 구분 없이)
    if filename.lower().endswith((".step", ".stp")):
        
        # 파일 경로 설정
        input_path = os.path.join(input_folder, filename)
        
        # 출력 파일명 설정 (확장자만 .stl로 변경)
        name_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{name_without_ext}.stl")
        
        try:
            print(f"변환 중: {filename} -> {name_without_ext}.stl")
            
            # STEP 파일 로드
            result = cq.importers.importStep(input_path)
            
            # STL로 내보내기 (정밀도를 조절하고 싶다면 tolerance 옵션 추가 가능)
            cq.exporters.export(result, output_path)
            
        except Exception as e:
            print(f"오류 발생 ({filename}): {e}")

print("\n모든 작업이 완료되었습니다.")