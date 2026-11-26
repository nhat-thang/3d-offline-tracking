"""Xem nhanh mot file ket qua Waymo (.bin, metrics_pb2.Objects): so box, so chuoi, so frame, theo lop, co tracking id khong.
Chay: python waymo/inspect_bin.py <file.bin>"""
import sys, collections
from waymo_open_dataset.protos import metrics_pb2

objs = metrics_pb2.Objects()
objs.ParseFromString(open(sys.argv[1], 'rb').read())
ctx, frames, types, ids = set(), set(), collections.Counter(), set()
for o in objs.objects:
    ctx.add(o.context_name); frames.add((o.context_name, o.frame_timestamp_micros))
    types[o.object.type] += 1
    if o.object.id: ids.add(o.object.id)
print(f'{sys.argv[1]}: {len(objs.objects)} box, {len(ctx)} chuoi, {len(frames)} frame')
print('theo lop (1=vehicle, 2=ped, 4=cyc):', dict(types))
print('so tracking id khac nhau:', len(ids))
o = objs.objects[0]
print('vi du:', o.context_name, o.frame_timestamp_micros, o.score, o.object.box)
