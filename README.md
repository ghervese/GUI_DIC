_*muDIC_GUI*_ 
-------------------------------------------------------------------------------------------------------------
muDIC_GUI is a GUI environment dedicated to Digital Image Correlation carried out with muDIC class: <https://linkinghub.elsevier.com/retrieve/pii/S2352711019301967>.  This class is called through Tab 4.
It is divide in 8 tabs. You can use them one after another, or indepently.
1. Tab 0 is informative. It provides all the features of the application. Some are not implemented at that moment. Especially for the half of tab 5 and the entire tabs 6 and 7.
2. Tab 1: It provides the possiblity to prepare a test that uses one or two camera that have the same characteristics.  
4. Tab 2: It provides a way to crop a set of images so that it is possible to get rid of some elements that are not interesting. The image are lighter to process.  
5. Tab 3: It provides the possibility to define the ROI on a set of images, prepare the Finite Element mesh and then carry out the calculation with muDIC solver.  
6. Tab 4: It provides some tools to visualize the fields that are the results of the DIC calculations. Some features are added to the original muDIC database of results, like the residue to figure out cracks.  
7. Tab 5: It provides the possiblity to extract and analyse a set of local displacements. It is possible to calculate their derivative by means of Finite Difference calculations. It is then possible to analyse as time histories for the time domain, Fast Fourier Transform for the frequency domain and also with Continuous Complex Cauchy Transform to visualize the results in time-frequency domain. It is possible to analyse that variations in graphics figuring out the variation of the apparent non linear modes versus another quantity of interest that is evolving in the same time.  
8. Tab 6: It provides the possiblity to prepare a .comm, .med and .export files that can be used by Code_Aster solver to simulate the response of the sample. it affords the possiblity of carrying out Finite Element Model Updating. The evolution of boundary condition in displacement VS time along segments of the sample are smoothed along the segments by reducing them on polynomial basis. The value are determined by least square minimization process at each instant.   
9. Tab 7: It provides the information related to the version. For the moment as it is not finished, the Version is 0.1. When it has reached the full completion of the tabs it will be the first version V1.0. The next versions will be pulled an another branch of the repository.

