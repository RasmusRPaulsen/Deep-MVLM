# Deep learning based 3D landmark placement
A tool for precisely placing 3D landmarks on 3D facial scans

## Citing Deep-MVLM

If you use Deep-MVLM in your research, please cite the
[paper](TBD):
```
@inproceedings{paulsen2018multi,
  title={Multi-view Consensus CNN for 3D Facial Landmark Placement},
  author={Paulsen, Rasmus R and Juhl, Kristine Aavild and Haspang, Thilde Marie and Hansen, Thomas and Ganz, Melanie and Einarsson, Gudmundur},
  booktitle={Asian Conference on Computer Vision},
  pages={706--719},
  year={2018},
  organization={Springer}
}
```

## How to use Deep-MVLM
### Rendering types
The type of 3D rendering used is specified in the **image_channels** setting. The options are:
- **geometry** pure geometry rendering without texture (1 image channel)
- **depth** depth rendering (the z-buffer) similar to range scanners like the Kinect (1 image channel)
- **RGB** texture rendering (3 image channels)
- **RGB+depth** texture plus depth rendering (3+1=4 image channels)
- **geometry+depth** geometry plus depth rendering (1+1=2 image channels)

## Team
Rasmus R. Paulsen and Kristine Aavild Juhl

## License
Deep-MVLM is released under the MIT license. See the LICENSE file for more details.
