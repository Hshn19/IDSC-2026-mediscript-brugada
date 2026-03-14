import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping for 1D CNN.
    Highlights which time regions in the ECG drove the classification.

    Target layer: last ConvBlock in the encoder (deepest features).
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.activations  = None
        self.gradients    = None
        self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().clone()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(self, signal_tensor):
        """
        Generate Grad-CAM heatmap for a single ECG sample.

        Args:
            signal_tensor : torch.Tensor of shape (1, 12, 1200)

        Returns:
            cam : np.array of shape (1200,) — importance per time step
                  normalized to [0, 1]
        """
        self.model.eval()
        signal_tensor = signal_tensor.clone().requires_grad_(True)

        logit = self.model(signal_tensor)
        self.model.zero_grad()
        logit.backward()

        # Global average pool the gradients over time dimension
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1).squeeze()  # (T,)
        cam = torch.relu(cam).numpy()

        # Upsample from feature map size back to 1200 samples
        cam_upsampled = np.interp(
            np.linspace(0, len(cam), 1200),
            np.arange(len(cam)),
            cam
        )

        # Normalize to [0, 1]
        cam_min, cam_max = cam_upsampled.min(), cam_upsampled.max()
        if cam_max - cam_min > 1e-8:
            cam_upsampled = (cam_upsampled - cam_min) / (cam_max - cam_min)
        else:
            cam_upsampled = np.zeros(1200)

        return cam_upsampled


def run_gradcam_analysis(model, dataset, indices, save_dir,
                          leads_to_plot=None, threshold=0.55):
    """
    Run Grad-CAM on selected samples and save overlay plots.

    Args:
        model         : trained BrugadaCNN
        dataset       : BrugadaDataset instance
        indices       : list of dataset indices to analyze
        save_dir      : folder to save figures
        leads_to_plot : list of lead indices to visualize (default: V1,V2,V3)
        threshold     : decision threshold for prediction label
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if leads_to_plot is None:
        leads_to_plot = [6, 7, 8]   # V1=6, V2=7, V3=8 in standard 12-lead

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    # Hook into the last conv block of the encoder
    target_layer = model.encoder[-1].block[-3]   # last Conv1d before GAP
    gradcam      = GradCAM1D(model, target_layer)

    model.eval()
    results = []

    for idx in indices:
        signal, label = dataset[idx]
        signal_tensor = signal.unsqueeze(0)   # (1, 12, 1200)

        # Get prediction
        with torch.no_grad():
            logit = model(signal_tensor)
            prob  = torch.sigmoid(logit).item()

        pred_label = 1 if prob >= threshold else 0
        true_label = int(label.item())

        # Generate CAM
        cam = gradcam.generate(signal_tensor)

        # Determine result type
        if true_label == 1 and pred_label == 1:
            result_type = "True Positive"
            color       = "green"
        elif true_label == 0 and pred_label == 0:
            result_type = "True Negative"
            color       = "steelblue"
        elif true_label == 1 and pred_label == 0:
            result_type = "False Negative"
            color       = "red"
        else:
            result_type = "False Positive"
            color       = "orange"

        # Plot
        n_leads  = len(leads_to_plot)
        fig, axes = plt.subplots(n_leads, 2, figsize=(16, 3.5 * n_leads),
                                  gridspec_kw={'width_ratios': [3, 1]})
        if n_leads == 1:
            axes = [axes]

        fig.suptitle(
            f"Sample {idx} | True: {'Brugada' if true_label else 'Normal'} | "
            f"Pred: {'Brugada' if pred_label else 'Normal'} "
            f"(prob={prob:.3f}) | {result_type}",
            fontsize=12, fontweight='bold', color=color
        )

        t = np.arange(1200) / 100   # time axis in seconds

        for row, lead_idx in enumerate(leads_to_plot):
            ax_sig = axes[row][0]
            ax_cam = axes[row][1]
            lead_signal = signal.numpy()[lead_idx]
            lead_name   = lead_names[lead_idx]

            # ECG signal with Grad-CAM overlay
            ax_sig.plot(t, lead_signal, color='black', lw=0.8, zorder=2)
            ax_sig.fill_between(t, lead_signal.min(), lead_signal.max(),
                                 alpha=cam * 0.5,
                                 color='crimson', zorder=1)
            ax_sig.set_ylabel(f'Lead {lead_name}\n(mV)', fontsize=9)
            ax_sig.grid(alpha=0.3)

            if row == 0:
                ax_sig.set_title('ECG Signal + Grad-CAM Overlay', fontsize=10)
            if row == n_leads - 1:
                ax_sig.set_xlabel('Time (s)')

            # CAM importance bar
            ax_cam.barh(t, cam, color='crimson', alpha=0.7, height=0.01)
            ax_cam.set_xlim(0, 1)
            ax_cam.set_ylabel('Time (s)', fontsize=8)
            ax_cam.set_xlabel('Importance', fontsize=8)
            if row == 0:
                ax_cam.set_title('CAM', fontsize=10)

        plt.tight_layout()
        save_path = f"{save_dir}/gradcam_sample{idx}_{result_type.replace(' ','_')}.png"
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        results.append({
            'idx': idx, 'true': true_label, 'pred': pred_label,
            'prob': prob, 'type': result_type
        })

    return results