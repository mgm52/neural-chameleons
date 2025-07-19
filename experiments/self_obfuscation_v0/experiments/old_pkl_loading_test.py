import pickle
from obf_reps.metrics.probes import LogisticRegression

class PrefixRenamingUnpickler(pickle.Unpickler):
    PREFIX_MAP = {
        "conditional_experiment.conditional_training": "experiments.self_obfuscation_v0.utils_training",
        "conditional_experiment": "experiments.self_obfuscation_v0",
    }

    def find_class(self, module, name):
        for old, new in self.PREFIX_MAP.items():
            if module == old or module.startswith(old + "."):
                module = module.replace(old, new, 1)
                break
        return super().find_class(module, name)

def load_pickle(path):
    with open(path, "rb") as fh:
        return PrefixRenamingUnpickler(fh).load()


if __name__ == "__main__":
    metric: LogisticRegression = load_pickle("experiments/self_obfuscation_main/outputs/probe_checkpoints/20250527_153915_gemma_2_9b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_bl_0.1_withbasetest_pretrainprobes/test_apple.pkl")
    print(metric)