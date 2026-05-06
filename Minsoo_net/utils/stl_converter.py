import trimesh

def scale_and_overwrite_mesh(file_path, scale_factor=0.01):
    # 1. 메쉬 로드
    mesh = trimesh.load(file_path)
    
    # 2. 스케일 적용 (0.001배)
    # apply_scale은 정점 좌표(vertices)를 직접 수정합니다.
    mesh.apply_scale(scale_factor)
    
    # 3. 동일한 경로로 저장 (바꿔치기)
    # export 시 파일 확장자에 맞춰 포맷이 자동 결정됩니다.
    mesh.export(file_path)
    print(f"Successfully scaled and saved: {file_path}")

# 사용 예시
file_path = "/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/example_model.obj" # 또는 .stl, .ply 등
scale_and_overwrite_mesh(file_path)