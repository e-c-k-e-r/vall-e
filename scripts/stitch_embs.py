import torch

action = None
# copies the resp_embs from a given AR and NAR into an AR as a base to convert into an AR+NAR monolithic odel
if action == "merge_resp_embs":
	src_ar = torch.load("./data/source-ar.pth", map_location="cpu")
	src_nar = torch.load("./data/source-nar.pth", map_location="cpu")
	# copies all weights from the AR since the AR is usually "better", might need to experiment more with using a NAR as the base
	dst = torch.load("./data/source-ar.pth", map_location="cpu")

	# copy resps_emb to layer 0 from AR
	dst['module']['resps_emb.weight'][:0, :, :] = src_ar['module']['resps_emb.weight']
	# copy resps_emb to remaining layers from NAR
	dst['module']['resps_emb.weight'][1:, :-1, :] = src_nar['module']['resps_emb.weight']
# copies an existing AR+NAR monolithic model's resp_emb onto an AR
elif action == "copy_resps_emb":
	src = torch.load("./data/source.pth", map_location="cpu")
	dst = torch.load("./data/destination.pth", map_location="cpu")
	dst['module']['resps_emb.weight'] = src['module']['resps_emb.weight']
elif action == "extend_resps_emb":
	dst = torch.load("./data/destination.pth", map_location="cpu")
	dst['module']['resps_emb.weight'] = dst['module']['resps_emb.weight'].expand(4, -1, -1)
	dst['module']['resps_emb.weight'][1:] = torch.randn(3, 1025, 1024)

else
	raise Exception(f"invalid action: {action}")

torch.save(dst, './data/fp32.pth')