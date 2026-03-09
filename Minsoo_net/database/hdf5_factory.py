"""
HDF5 직렬화/역직렬화 팩토리
- dex-net 원본에서 Minsoo_net용으로 수정
- meshpy, autolab_core, perception 의존성 제거
- SE(3)는 4x4 numpy 행렬 사용
- 렌더링 관련 코드 제거 (Isaac Sim 사용)
"""
import datetime as dt
import logging
import numpy as np

from Minsoo_net.database.keys import *
from Minsoo_net.grasp.grasp import ParallelJawGrasp


class Hdf5ObjectFactory(object):
    """HDF5 그룹/데이터셋을 Python 객체로 변환하는 팩토리 클래스"""

    # ── SDF ──────────────────────────────────────────────

    @staticmethod
    def sdf_3d(data):
        """HDF5 → SDF dict"""
        return {
            'data': np.array(data[SDF_DATA_KEY]),
            'origin': np.array(data.attrs[SDF_ORIGIN_KEY]),
            'resolution': float(data.attrs[SDF_RES_KEY]),
        }

    @staticmethod
    def write_sdf_3d(sdf, data):
        """SDF dict → HDF5"""
        data.create_dataset(SDF_DATA_KEY, data=sdf['data'])
        data.attrs.create(SDF_ORIGIN_KEY, sdf['origin'])
        data.attrs.create(SDF_RES_KEY, sdf['resolution'])

    # ── Mesh ─────────────────────────────────────────────

    @staticmethod
    def mesh_3d(data):
        """HDF5 → mesh dict"""
        result = {
            'vertices': np.array(data[MESH_VERTICES_KEY]),
            'triangles': np.array(data[MESH_TRIANGLES_KEY]),
        }
        if MESH_NORMALS_KEY in data.keys():
            result['normals'] = np.array(data[MESH_NORMALS_KEY])
        return result

    @staticmethod
    def write_mesh_3d(mesh, data):
        """mesh dict → HDF5"""
        data.create_dataset(MESH_VERTICES_KEY, data=mesh['vertices'])
        data.create_dataset(MESH_TRIANGLES_KEY, data=mesh['triangles'])
        if 'normals' in mesh and mesh['normals'] is not None:
            data.create_dataset(MESH_NORMALS_KEY, data=mesh['normals'])

    # ── Stable Poses ─────────────────────────────────────

    @staticmethod
    def stable_poses(data):
        """HDF5 → list of stable pose dicts"""
        num_stable_poses = data.attrs[NUM_STP_KEY]
        stable_poses = []
        for i in range(num_stable_poses):
            stp_key = POSE_KEY + '_' + str(i)
            p = float(data[stp_key].attrs[STABLE_POSE_PROB_KEY])
            r = np.array(data[stp_key].attrs[STABLE_POSE_ROT_KEY])
            try:
                x0 = np.array(data[stp_key].attrs[STABLE_POSE_PT_KEY])
            except KeyError:
                x0 = np.zeros(3)

            # SE(3) 4x4 행렬로 구성
            T = np.eye(4)
            T[:3, :3] = r
            T[:3, 3] = x0

            stable_poses.append({
                'id': stp_key,
                'p': p,
                'r': r,
                'x0': x0,
                'T': T,
            })
        return stable_poses

    @staticmethod
    def stable_pose(data, stable_pose_id):
        """HDF5 → single stable pose dict"""
        p = float(data[stable_pose_id].attrs[STABLE_POSE_PROB_KEY])
        r = np.array(data[stable_pose_id].attrs[STABLE_POSE_ROT_KEY])
        try:
            x0 = np.array(data[stable_pose_id].attrs[STABLE_POSE_PT_KEY])
        except KeyError:
            x0 = np.zeros(3)

        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = x0

        return {
            'id': stable_pose_id,
            'p': p,
            'r': r,
            'x0': x0,
            'T': T,
        }

    @staticmethod
    def write_stable_poses(stable_poses, data, force_overwrite=False):
        """list of stable pose dicts → HDF5"""
        num_stable_poses = len(stable_poses)
        data.attrs.create(NUM_STP_KEY, num_stable_poses)
        for i, sp in enumerate(stable_poses):
            stp_key = POSE_KEY + '_' + str(i)
            if stp_key not in data.keys():
                data.create_group(stp_key)
                data[stp_key].attrs.create(STABLE_POSE_PROB_KEY, sp['p'])
                data[stp_key].attrs.create(STABLE_POSE_ROT_KEY, sp['r'])
                data[stp_key].attrs.create(STABLE_POSE_PT_KEY, sp['x0'])
            elif force_overwrite:
                data[stp_key].attrs[STABLE_POSE_PROB_KEY] = sp['p']
                data[stp_key].attrs[STABLE_POSE_ROT_KEY] = sp['r']
                data[stp_key].attrs[STABLE_POSE_PT_KEY] = sp['x0']
            else:
                logging.warning('Stable %s already exists and overwrite was not requested.' % stp_key)
                return None

    # ── Grasps ───────────────────────────────────────────

    @staticmethod
    def grasps(data):
        """HDF5 → list of ParallelJawGrasp"""
        grasps = []
        num_grasps = data.attrs[NUM_GRASPS_KEY]
        for i in range(num_grasps):
            grasp_key = GRASP_KEY + '_' + str(i)
            if grasp_key in data.keys():
                grasp_id = int(data[grasp_key].attrs[GRASP_ID_KEY])
                center = np.array(data[grasp_key].attrs[GRASP_CENTER_KEY])
                axis = np.array(data[grasp_key].attrs[GRASP_AXIS_KEY])
                open_width = float(data[grasp_key].attrs[GRASP_WIDTH_KEY])

                approach = None
                if GRASP_APPROACH_KEY in data[grasp_key].attrs:
                    approach = np.array(data[grasp_key].attrs[GRASP_APPROACH_KEY])

                g = ParallelJawGrasp(
                    center=center,
                    axis=axis,
                    open_width=open_width,
                    approach=approach,
                )
                g.id = grasp_id
                grasps.append(g)
            else:
                logging.debug('Grasp %s is corrupt. Skipping' % grasp_key)
        return grasps

    @staticmethod
    def write_grasps(grasps, data, force_overwrite=False):
        """list of ParallelJawGrasp → HDF5"""
        num_grasps = data.attrs[NUM_GRASPS_KEY]
        num_new_grasps = len(grasps)

        dt_now = dt.datetime.now()
        creation_stamp = '%s-%s-%s-%sh-%sm-%ss' % (
            dt_now.month, dt_now.day, dt_now.year,
            dt_now.hour, dt_now.minute, dt_now.second,
        )

        for i, grasp in enumerate(grasps):
            grasp_id = getattr(grasp, 'id', None)
            if grasp_id is None:
                grasp_id = i + num_grasps
            grasp_key = GRASP_KEY + '_' + str(grasp_id)

            if grasp_key not in data.keys():
                data.create_group(grasp_key)
                data[grasp_key].attrs.create(GRASP_ID_KEY, grasp_id)
                data[grasp_key].attrs.create(GRASP_CENTER_KEY, grasp.center)
                data[grasp_key].attrs.create(GRASP_AXIS_KEY, grasp.axis)
                data[grasp_key].attrs.create(GRASP_WIDTH_KEY, grasp.open_width)
                if grasp.approach is not None:
                    data[grasp_key].attrs.create(GRASP_APPROACH_KEY, grasp.approach)
                data[grasp_key].create_group(GRASP_METRICS_KEY)
            elif force_overwrite:
                data[grasp_key].attrs[GRASP_ID_KEY] = grasp_id
                data[grasp_key].attrs[GRASP_CENTER_KEY] = grasp.center
                data[grasp_key].attrs[GRASP_AXIS_KEY] = grasp.axis
                data[grasp_key].attrs[GRASP_WIDTH_KEY] = grasp.open_width
                if grasp.approach is not None:
                    if GRASP_APPROACH_KEY in data[grasp_key].attrs:
                        data[grasp_key].attrs[GRASP_APPROACH_KEY] = grasp.approach
                    else:
                        data[grasp_key].attrs.create(GRASP_APPROACH_KEY, grasp.approach)
            else:
                logging.warning('Grasp %d already exists and overwrite was not requested.' % grasp_id)
                return None

        data.attrs[NUM_GRASPS_KEY] = num_grasps + num_new_grasps
        return creation_stamp

    # ── Grasp Metrics ────────────────────────────────────

    @staticmethod
    def grasp_metrics(grasps, data):
        """HDF5 → dict of grasp metrics"""
        grasp_metrics = {}
        for grasp in grasps:
            grasp_id = grasp.id
            grasp_key = GRASP_KEY + '_' + str(grasp_id)
            grasp_metrics[grasp_id] = {}
            if grasp_key in data.keys():
                grasp_metric_data = data[grasp_key][GRASP_METRICS_KEY]
                for metric_name in grasp_metric_data.attrs.keys():
                    grasp_metrics[grasp_id][metric_name] = grasp_metric_data.attrs[metric_name]
        return grasp_metrics

    @staticmethod
    def write_grasp_metrics(grasp_metric_dict, data, force_overwrite=False):
        """dict of grasp metrics → HDF5"""
        for grasp_id, metric_dict in grasp_metric_dict.items():
            grasp_key = GRASP_KEY + '_' + str(grasp_id)
            if grasp_key in data.keys():
                grasp_metric_data = data[grasp_key][GRASP_METRICS_KEY]
                for metric_tag, metric in metric_dict.items():
                    if metric_tag not in grasp_metric_data.attrs.keys():
                        grasp_metric_data.attrs.create(metric_tag, metric)
                    elif force_overwrite:
                        grasp_metric_data.attrs[metric_tag] = metric
                    else:
                        logging.warning(
                            'Metric %s already exists for grasp %s.' % (metric_tag, grasp_id)
                        )
                        return False
        return True

    # ── Connected Components ─────────────────────────────

    @staticmethod
    def connected_components(data):
        """HDF5 → dict of mesh dicts"""
        if CONNECTED_COMPONENTS_KEY not in data.keys():
            return None
        out = {}
        for key in data[CONNECTED_COMPONENTS_KEY]:
            out[key] = Hdf5ObjectFactory.mesh_3d(data[CONNECTED_COMPONENTS_KEY][key])
        return out

    @staticmethod
    def write_connected_components(connected_components, data, force_overwrite=False):
        """list of mesh dicts → HDF5"""
        if CONNECTED_COMPONENTS_KEY in data.keys():
            if force_overwrite:
                del data[CONNECTED_COMPONENTS_KEY]
            else:
                logging.warning('Connected components already exist, aborting')
                return False
        cc_group = data.create_group(CONNECTED_COMPONENTS_KEY)
        for idx, mesh in enumerate(connected_components):
            one_cc_group = cc_group.create_group(str(idx))
            Hdf5ObjectFactory.write_mesh_3d(mesh, one_cc_group)
        return True
