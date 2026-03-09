"""
HDF5 데이터베이스 키 상수 정의
- dex-net 원본에서 Minsoo_net용으로 수정
- ParallelJawGrasp 구조에 맞게 grasp 키 변경
- 렌더링 관련 키 제거 (Isaac Sim 사용)
"""

# ── 최상위 구조 ──────────────────────────────────────
DATASETS_KEY = 'datasets'
DATASET_KEY = 'dataset'
CREATION_KEY = 'time_created'
OBJECTS_KEY = 'objects'
METRICS_KEY = 'metrics'
CATEGORY_KEY = 'category'
MASS_KEY = 'mass'

# ── Mesh ─────────────────────────────────────────────
MESH_KEY = 'mesh'
MESH_VERTICES_KEY = 'vertices'
MESH_TRIANGLES_KEY = 'triangles'
MESH_NORMALS_KEY = 'normals'

# ── SDF ──────────────────────────────────────────────
SDF_KEY = 'sdf'
SDF_DATA_KEY = 'data'
SDF_ORIGIN_KEY = 'origin'
SDF_RES_KEY = 'resolution'

# ── Stable Poses ─────────────────────────────────────
STP_KEY = 'stable_poses'
NUM_STP_KEY = 'num_stable_poses'
POSE_KEY = 'pose'
STABLE_POSE_PROB_KEY = 'p'
STABLE_POSE_ROT_KEY = 'r'
STABLE_POSE_PT_KEY = 'x0'

# ── Grasps (ParallelJawGrasp 구조) ───────────────────
GRASPS_KEY = 'grasps'
NUM_GRASPS_KEY = 'num_grasps'
GRASP_KEY = 'grasp'
GRASP_ID_KEY = 'id'
GRASP_CENTER_KEY = 'center'
GRASP_AXIS_KEY = 'axis'
GRASP_WIDTH_KEY = 'open_width'
GRASP_APPROACH_KEY = 'approach'
GRASP_METRICS_KEY = 'metrics'

# ── Connected Components ─────────────────────────────
CONNECTED_COMPONENTS_KEY = 'connected_components'
CONVEX_PIECES_KEY = 'convex_pieces'

# ── Metadata ─────────────────────────────────────────
METADATA_KEY = 'metadata'
METADATA_TYPE_KEY = 'type'
METADATA_DESC_KEY = 'description'
