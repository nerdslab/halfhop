# Half-Hop
Official Implementation of Half-Hop: A graph upsampling approach for slowing down message passing [Paper](https://openreview.net/forum?id=lXczFIwQkv)

Half-Hop is plug-and-play, and works with a wide range of datasets, architectures, and learning objectives!

![](overview.png)


Example usage:
```python3
from halfhop import HalfHop
# apply augmentation
transform = HalfHop(alpha=0.5)
data = transform(data)

# feedforward
y = model(data)

# get rid of slow nodes 
y = y[~data.slow_node_mask]
```

If you find the code useful for your research, please consider citing our work:
```
@article{azabou2023half,
  title={Half-Hop: A graph upsampling approach for slowing down message passing},
  author={Azabou, Mehdi and Ganesh, Venkataramana and Thakoor, Shantanu and Lin, Chi-Heng and Sathidevi, Lakshmi and Liu, Ran and Valko, Michal and Veli{\v{c}}kovi{\'c}, Petar and Dyer, Eva L},
  year={2023}
}
```
