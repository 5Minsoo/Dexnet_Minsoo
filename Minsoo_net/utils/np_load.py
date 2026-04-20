# mark_all_done.py
import zarr, yaml

with open('/home/minsoo/Dexnet_Minsoo/Minsoo_net/config/master_config.yaml') as f:
    zarr_path = yaml.safe_load(f).get("zarr_path", "grasp_dataset_big1.zarr")

store = zarr.open(zarr_path, mode='a')

print(len(store.keys))

print(f"\n총 {len(list(store.keys()))}개 물체에 done 마커 추가")