from Minsoo_net.grasp import GraspPipeline
p=GraspPipeline('/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/object.stl',30,0.0,0.012,30)
gen=    p.execute(use_visual=True,start_index=8)
for i in range(30):
    try:
        result = next(gen) # 하나씩 가져오기
    except StopIteration:
        break