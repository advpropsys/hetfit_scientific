# hetfit_scientific
Data science code for Hall effect thrusters

# Abstract:
  Hall effect thrusters are one of the most versatile and
  popular electric propulsion systems for space use. Industry trends
  towards interplanetary missions arise advances in design development
  of such propulsion systems. It is understood that correct sizing of
  discharge channel in Hall effect thruster impact performance greatly.
  Since the complete physics model of such propulsion system is not yet
  optimized for fast computations and design iterations, most thrusters
  are being designed using so-called scaling laws. But this work focuses
  on rather novel approach, which is outlined less frequently than
  ordinary scaling design approach in literature. Using deep machine
  learning it is possible to create predictive performance model, which
  can be used to effortlessly get design of required hall thruster with
  required characteristics using way less computing power than design
  from scratch and way more flexible than usual scaling approach.
author: Korolev K.V [^1]
title: Hall effect thruster design via deep neural network for additive
  manufacturing

# Nomenclature

<div class="longtable*" markdown="1">

$U_d$ = discharge voltage  
$P$ = discharge power  
$T$ = thrust  
$\dot{m}_a$ = mass flow rate  
$I_{sp}$ = specific impulse  
$\eta_m$ = mass utilization efficiency  
$\eta_a$ = anode efficiency  
$j$ = $P/v$ \[power density\]  
$v$ = discharge channel volume  
$h, d, L$ = generic geometry parameters  
$C_*$ = set of scaling coefficients  
$g$ = free-fall acceleration  
$M$ = ion mass

</div>

# Introduction

<span class="lettrine">T</span><span class="smallcaps">he</span>
application of deep learning is extremely diverse, but in this study it
focuses on case of hall effect thruster design. Hall effect thruster
(HET) is rather simple DC plasma acceleration device, due to complex and
non linear process physics we don’t have any full analytical performance
models yet. Though there are a lot of ways these systems are designed in
industry with great efficiencies, but in cost of multi-million research
budgets and time. This problem might be solved using neural network
design approach and few hardware iteration tweaks(Plyashkov et al.
2022-10-25).

Scaled thrusters tend to have good performance but this approach isn’t
that flexible for numerous reasons: first and foremost, due to large
deviations in all of the initial experimental values accuracy can be not
that good, secondly, it is hardly possible to design thruster with
different power density or $I_{sp}$ efficiently.

On the other hand, the neural network design approach has accuracy
advantage only on domain of the dataset(Plyashkov et al. 2022-10-25),
this limitations is easily compensated by ability to create relations
between multiple discharge and geometry parameters at once. Hence this
novel approach and scaling relations together could be an ultimate
endgame design tool for HET.

Note that neither of these models do not include cathode efficiencies
and performances. So as the neutral gas thrust components. Most
correlations in previous literature were made using assumption or physics laws(Shagayda and Gorshkov 2013-03), in this paper the new
method based on feature generation, GAN dataset augmentation and ML
feature selection is suggested.





Normalizing flows can be used for generating new samples for sensor or engineering data by modeling the distribution of the data using a series of invertible transformations. These transformations can capture the complex probability density function of the data, and the resulting model can be used to generate new samples.

For example, if we have a dataset of sensor readings from a machine, we can use normalizing flows to model the distribution of these readings. We can then generate new samples of sensor readings that are similar to the original dataset, but with some variation. This can be useful for testing machine learning algorithms, evaluating sensor performance, or generating synthetic data for training new models.

Normalizing flows can also be used for anomaly detection in sensor data. By learning the distribution of normal sensor readings, we can identify instances where the data deviates significantly from this normal distribution, indicating a potential anomaly or fault in the machine.

Overall, normalizing flows provide a powerful tool for generating new samples of sensor or engineering data, as well as for analyzing and understanding the underlying distribution of the data.

Full demo and docs: https://advpropsys.streamlit.app



