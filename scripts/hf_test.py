import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# tokenizer = LlamaTokenizer.from_pretrained("./training/llama-encodec-ar+nar-len/hf/")
model = LlamaForCausalLM.from_pretrained("./training/llama-encodec-ar+nar-len/hf/")

phns = [1,85,4,128,26,4,186,4,89,33,25,4,48,4,134,25,52,86,4,34,97,27,11,2]
proms = [
	[780,835,835,835,339,395,798,537,537,537,537,222,76,989,548,65,705,375,261,375,297,503,529,571,707,346,464,862,148,496,574,115,115,438,934,339,865,876,63,40,779,461,602,794,10,220,398,869,639,705,869,917,705,893,215,705,869,938,439,175,139,506,375,529,297,705,651,238,962,461,195,441,377,581,473,795,644,626,459,981,767,670,696,73,779,257,408,1017,1019,133,133,1017,835,604,699,626,67,92,707,92,179,179,772,869,441,799,917,238,745,904,904,904,106,133,1019,1017,1017,395,883,87,519,594,1002,682,996,540,186,1019,430,202,347,889,61,92,542,297,67,669,571,707,346,67,359,571,707,669,604,25,1008,810,35,621,67,600,333,123,284,568,817,243,778,464,638,610,359,538,464,975,321,700,377,484,179,284,284,621,538,464,745,171,171,159,744,159,287,461,69,15,529,67,92,669,464,515,605,24,822,865,293,62,172,638,359,562,138,839,846,775,556,688,1006,917,297,312,148,331,496,646,67,314,15,705,131,855,662,287,172,85,538,519,762,450,391,609,643,778,80,287,794,794,115,785,794,461,699,519,932,522,652,262,508,902,932,932,391,769,18,507,90,442,762,610,610,669,605,310,855,56,989,863,195,464,604,257,904,632,786,951,461,239,195,878,771,146,481,146,481,434,643,917,280,67,464,115,744,744,115,115,115,819,709,63,368,359,519,996,616,464,996,616,519,762,917,841,772,568,954,600,422,893,592,464,626,86,143,615,171,744,744,196,115,821,415,521,799,654,839,644,473,592,953,523,855,738,855,876,876,1017,63,329]
]
sep = [17685]
rvq_lvl = [17666]
lang = [17686]
len_seq = [17674]

for i, t in enumerate( proms[0] ):
	proms[0][i] = t + 256 + 1024

ids = torch.tensor(phns + sep + lang + sep + rvq_lvl + sep + proms[0] + sep + len_seq, device="cuda", dtype=torch.int32)
pos_ids = torch.tensor( [*range(len(phns)+1)] + [*range(2)] + [*range(2)] + [*range(len(proms[0])+1)] + [0], device="cuda", dtype=torch.int32)

start = 17674 # 8448
end = start + 10 # 1025

with torch.no_grad():
	original_lm_head = model.lm_head.weight

	model.lm_head = torch.nn.Linear(1024, end - start, bias=False)
	model.lm_head.weight.copy_(original_lm_head[start:end])

model.to(device="cuda", dtype=torch.float16)
model.eval()

n_decoded = 0
while True:
	out = model(input_ids=ids.unsqueeze(0), position_ids=pos_ids.unsqueeze(0))

	#logits = out.logits[0, -1:, start:end]
	logits = out.logits[0, -1:, :]
	tokens = logits.argmax(dim=-1)
	n_decoded += 1

	print( n_decoded, tokens )

	if end in tokens or n_decoded > 5:
		break

	ids = torch.concat( [ ids, tokens + start ] )
	pos_ids = torch.concat( [ pos_ids, torch.tensor([n_decoded]).to(pos_ids) ] )

print( out )
print( ids )