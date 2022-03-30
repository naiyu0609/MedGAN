# MedGAN (Pytorch)  
MRI影像有各種成像方式，若要分析T1時間常數就要靠T1 mapping，T1 mapping可以直接對組織的T1進行定量測量。而T1 mapping技術的原理是收集心動週期一系列不同反轉時間點的影像，測量每個組織(像素)的T1值進行診斷，透過T1值的變異可以診斷出疾病的病變過程，在心臟疾病診斷方面有著一定的優勢，倘若能利用AI自動化分割出心肌位置將會加速醫生診斷。  
![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/tra-pass.jpg)  
利用傳統算法逐點進行擬和找出T1值會有一定的雜訊產生，而再透過擬和公式找出心室排空血液時間點的影像，會有更多的雜訊產生(如上圖右側)，因此如果能透過GAN的方式產生心室中完全排空血液的影像(如下圖右側)，將會有助於提升心肌便是準確度。  
![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/gan-pass.jpg)  

使用MedGAN: Medical image translation using GANs[2]此篇論文為架構加以修改
![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/架構.JPG)  
![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/block構架.jpg)  

# Result
| T1map | synthesis by GAN | synthesis by function |
|:----------:|:----------:|:----------:|
|![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/t1map1.png)|![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/gan1.png)||![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/syn1.png)|  
|![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/t1map2.png)|![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/gan2.png)||![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/syn2.png)|  
|![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/t1map3.png)|![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/gan3.png)||![image](https://github.com/naiyu0609/MedGAN/blob/main/jpg/syn3.png)|  

# References
[1] https://github.com/milesial/Pytorch-UNet  
[2] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
[3] K. Armanious et al., "MedGAN: Medical image translation using GANs", Comput. Med. Imag. Graph., vol. 79, Jan. 2020.
