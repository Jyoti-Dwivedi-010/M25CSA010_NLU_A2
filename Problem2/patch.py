# This patch adds a new section to the `train_evaluate.py` file that trains each of the models defined in `models.py`, evaluates their performance based on diversity and novelty metrics, and generates sample names. Additionally, it plots the training loss curves for each model and saves the plot as an image file. The results of the evaluation are printed in a formatted report. 
text = open('train_evaluate.py', 'r', encoding='utf-8').read()
idx = text.find('    results = {}')
new_text = text[:idx] + '''    results = {}
    all_losses = {}
    for name, model in models_dict.items():
        trained_model, epoch_losses = train_model(model, names, char2idx, epochs=10, lr=0.005, save_name=name)
        all_losses[name] = epoch_losses
        m_diversity, m_novelty, samples = evaluate(trained_model, names, char2idx, idx2char, num_samples=100, device=device)
        results[name] = {
            "diversity": m_diversity,
            "novelty": m_novelty,
            "samples": samples
        }
    plt.figure(figsize=(10, 6))
    for model_name, losses in all_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=model_name, marker="o")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    print("\\nSaved training loss curves to loss_curves.png")
    print("\\n\\n=== EVALUATION REPORT ===")
    for name, res in results.items():
        print(f"\\nModel: {name}")
        print("Diversity Rate: {:.2%}".format(res["diversity"]))
        print("Novelty Rate:   {:.2%}".format(res["novelty"]))
        print(f"Sample Names generated: {', '.join(res['samples'])}")
'''
open('train_evaluate.py', 'w', encoding='utf-8').write(new_text)
print("done")
