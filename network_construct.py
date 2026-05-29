from embed_EGA import *
import json

def construct(fwd_path, back_path):
    vec = vectorizer()
    ega = TMFG()
    uva = UVA()
    bootega = BootEGA()

    # Load forward translation embeddings (original + translation)
    fwd_embeds = vec.process_json(fwd_path, batch_size=32)
    orig_embedding = fwd_embeds['original']['embeddings']
    fwd_embedding = fwd_embeds['translation']['embeddings']
    item_numbers = fwd_embeds['item_numbers']

    # Load back translation embeddings
    back_embeds = vec.process_json(back_path, batch_size=32)
    back_embedding = back_embeds['translation']['embeddings']

    data_orig = orig_embedding.T
    data_fwd = fwd_embedding.T
    data_back = back_embedding.T

    corr_orig = ega.compute_corr(orig_embedding)
    corr_fwd = ega.compute_corr(fwd_embedding)
    corr_back = ega.compute_corr(back_embedding)

    idx = [range(13)]
    print(idx)

    mem_a, _ = ega.calculate_membership(corr_orig)
    mem_b, _ = ega.calculate_membership(corr_fwd)
    mem_c, _ = ega.calculate_membership(corr_back)

    for i in range(4):
        print(f"Visualizing Original Network number{i}")
        ega.visualize_network(
            corr_matrix=corr_orig, membership=mem_a, item_labels=idx,
            output_file=f"orig_network_{i}.png",
            title='Original TMFG Network',
            node_size=8, label_size=0.8
        )
        print(f"Visualizing Forward Network number{i}")
        ega.visualize_network(
            corr_matrix=corr_fwd, membership=mem_b, item_labels=idx,
            output_file=f"fwd_network_{i}.png",
            title='Forward Translation TMFG Network',
            node_size=8, label_size=0.8
        )
        print(f"Visualizing Backward Network number{i}")
        ega.visualize_network(
            corr_matrix=corr_back, membership=mem_c, item_labels=idx,
            output_file=f"back_network_{i}.png",
            title='Backward Translation TMFG Network',
            node_size=8, label_size=0.8
        )
        i+=1
    print("Visualization Completed")

if __name__ == "__main__":
    fwd_file = "results/fwd_gpt_t0.8.json"
    back_file = "results/back_fwd_gpt_t0.8__back_claude_t0.8.json"
    res = construct(fwd_file, back_file)