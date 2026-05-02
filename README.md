# 🌐 Computable Compute-Power Graph (CCPG)
**Formal Theory, Empirical Validation, and Tractability Analysis for Dynamic Heterogeneous Compute Scheduling**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19911961.svg)](https://doi.org/10.5281/zenodo.19911961) [![arXiv](https://img.shields.io/badge/arXiv-2X04.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2X04.XXXXX) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status:** Initial Theoretical Release & Core Engine Demo (April 2026)  
> **Authors:** Forrest (Zhang ZhiPing) | Beijing Zhongda Education Technology Research Institute  

## 📖 Abstract

**CCPG** is a foundational framework for next-generation decentralized and heterogeneous computing. It transforms physical infrastructure—GPUs, TPUs, NPU clusters, and optical interconnects—into a **formally defined, computable topological network**. 

By unifying computational capacity, memory, network latency, power cost, and thermal state into a single optimization problem, CCPG shifts infrastructure management from static orchestration (e.g., Kubernetes Best-Fit) to **Cognitive Adaptive Computing**.

## 🧠 Core Theoretical Innovation

The principal value of CCPG lies in the mathematical unification of five historically siloed constraint dimensions:

1. **Computational Capacity ($C_i$):** Real-time TFLOPS availability.
2. **Memory ($M_i$):** Volatile VRAM/DRAM state.
3. **Network Latency ($L_{ij}$):** Dynamic topology-aware delay.
4. **Power Cost ($P_i$):** Economic and energy efficiency.
5. **Thermal State ($\Theta_i$):** Thermodynamic hardware constraints.

This integration enables the system to solve for the **Global Utility Function**:

$$J = \lambda_1 \sum_{i \in V} z_i P_i + \lambda_2 \sum_{(i,j) \in E} \mathbf{A}_{ij} L_{ij} + \lambda_3 \mathcal{H}(G)$$

## 🚀 Features & Implementation

This repository contains the `ccpg-core` Python demo, which implements the following core theoretical mechanisms:

* **Heterogeneous Graph Instantiation:** Utilizes `NetworkX` to construct a physical topology comprising GPU servers (nodes) and InfiniBand/fiber-optic links (edges).
* **Tensorization:** Converts node compute power, VRAM, and electricity costs, alongside edge bandwidth and latency, into `PyTorch` Tensors. This establishes the foundation for future integration with Graph Neural Networks (GNNs).
* **Composite Cost Function:** Leverages PyTorch's tensor parallel processing to dynamically calculate real-time cost weights for every compute-power link based on the $J$ formulation.
* **Resource-Constrained Optimization:** Identifies the optimal compute-delivery path (Constrained Shortest Path) with the lowest cost and minimum latency, strictly adhering to memory and Flops thresholds.

## 📊 Empirical Validation (Simulation)

Based on statistical parameters derived from **Google Cluster Trace v3 (GCT-v3)**, our simulation (500 jobs, 48-node heterogeneous cluster) demonstrates significant improvements over baseline schedulers:

| Scheduler | Avg JCT (s) | GPU Util. | Success Rate |
| :--- | :--- | :--- | :--- |
| Kubernetes Best-Fit | 2382.8 | 35.12% | 79.80% |
| DRF | 2227.0 | 40.43% | 87.00% |
| **CCPG (This Work)** | **2013.5** | **40.78%** | **80.40%** |
| *CCPG vs K8s-BF* | *-15.5%* | *+16.1%* | *+0.8%* |

*(Note: Negative percentages for JCT indicate improvement.)*

## 💻 Quick Start & Installation

### Requirements
* Python 3.8+
* PyTorch >= 2.0.0
* NetworkX >= 2.8
* 双重授权（Dual-License）声明”：

"The ER-CCPG core engine and related scheduling plugins are licensed under the AGPL v3.0. For enterprise use cases (e.g., integration into proprietary cloud infrastructure or closed-source commercial products) where the AGPL v3.0 is not applicable, please contact the author to purchase a Commercial License."
(ER-CCPG 核心引擎及相关调度插件采用 AGPL v3.0 协议开源。对于无法适用 AGPL v3.0 的企业级使用场景（例如集成到专有云基础设施或闭源商业产品中），请联系作者购买商业授权。)

### Installation
Clone the repository and install dependencies:
```bash
git clone [https://github.com/forrestzzp/CCPG.git](https://github.com/forrestzzp/CCPG.git)
cd ccpg-core
pip install -r requirements.txt
