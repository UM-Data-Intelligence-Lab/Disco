# Disco

Disco learns disentangled multimodal region representations from heterogeneous urban data for urban region prediction. It organizes modality-specific representations into visual and functional contexts, separates unique and shared information, and identifies cross-region commonality through cosine similarity in the shared feature space. An adaptive Mixture-of-Experts (MoE) module then dynamically fuses the disentangled representations for downstream prediction and interpretable contribution analysis.

Please see the details in our paper below:

- *Disentangled Multimodal Region Representation Learning with Cross-Region Commonality Mining*, accepted by IEEE Transactions on Mobile Computing (TMC).

## Requirements

The code requires the following environment and package versions:

```text
python==3.10.20
numpy==2.2.6
scipy==1.15.3
pandas==2.3.3
scikit-learn==1.7.2
joblib==1.5.3
threadpoolctl==3.6.0
torch==2.12.0
torchvision==0.27.0
networkx==3.4.2
python-louvain==0.16
h3==4.5.0
shapely==2.1.2
Pillow==12.2.0
geopandas==1.1.3
rasterio==1.4.4
```

### Usage
To reproduce the experiments in the paper, please follow the three stages below in order: IntraModality training, DisentanglePositive training, and DownstreamTask evaluation.

For the IntraModality training phase, run the following command with your own data:

```bash
bash intramodality.sh
```

For the DisentanglePositive training phase, run the following command with your own data:

```bash
bash disentanglepositive.sh
```

For the DownstreamTask evaluation phase, run the following command with your own data:

```bash
bash downstreamtask_evaluate.sh
```

## Reference

If you use our code, please cite:

```bibtex

```

## Note

