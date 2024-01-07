#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
import tkinter.font as font
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
from tkinter import filedialog
import os
from matplotlib.patches import Rectangle
import muDIC as dic
from PIL import Image, ImageDraw
from matplotlib import image as mpimg
from matplotlib.pyplot import connect
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from matplotlib.widgets  import RectangleSelector
import matplotlib.collections as collections
from pathlib import Path
#######################################################################################################################################
#                                            User friendly GUI for Digital Image Correlation with muDIC   
#                                    V0.1 by Guillaume Hervé-Secourgeon // guillaume.herve-secourgeon[at]edf.fr
#######################################################################################################################################
# The purpose of this Python class is to provide an environment easing the use of muDIC and the pre-processing of a set of images
class muDIC_GUI:
    """
    This class is a GUI dedicated to 3 purposes:
    - Pre-process a set of experimental images
    - Carry out Digital Image Correlation based on muDIC class.
    - Post-process the results
    - Carry out comparisons with FEM computations results

    This class uses tkinter that is coming naturally with Python installations.
    The core of the class relies on muDIC class.
    Caution: for Ubuntu environment the muDIC class has to be installed manually and not through pip
    For theoretical details regarding muDIC see: https://www.sciencedirect.com/science/article/pii/S2352711019301967 
    (muDIC: An open-source toolkit for digital image correlation, S.N. Olufsen et al., SoftwareX Volume 11, January–June 2020, 100391)
    muDIC is a non-local FE based Digital Image Correlation class. It is entirely developed in Python.
    The other prerequisites are PIL, Numpy and MatPlotLib.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quantitative Analysis of Experimental Images with "+u"\N{GREEK SMALL LETTER MU}"+"DIC Class")
        self.fig_photo_first = None
        self.source_selection_path = None
        self.FEM_fig_photo_first = None
        self.mudicgui_version = 'v 0.1'
        self.DIC_settings = None
        self.DIC_results = None
        self.ax_sketch = None
        self.Figurecanvas_fig_setup_sketch = None
    plt.rcParams['figure.figsize'] = [0.1, 0.1]

    def Close(self):
        """
        Close the GUI and the app
        """
        self.root.destroy()    
    def run(self):
        """
        Launch the GUI
        This method initiates the principal loop of the tkinter based GUI and the app on screen.

        """
        self.create_gui()
        self.root.mainloop()

    def select_images_folder(self):
        """
        This method is a dialog box to get the path accessing to the input images
        """

        self.source_selection_path = filedialog.askdirectory(title='Select the Directory hosting the images')
        self.nb_files_max = len([entry for entry in os.listdir(self.source_selection_path)])
        print('nbre de fichiers',self.nb_files_max)


    def select_first_image(self):
        """
        This method is a dialog box to get the first input images of the stack of images
        """

        self.first_image_file = filedialog.askopenfilename(title='Select the first image of the stack of images',initialdir=self.source_selection_path)
        self.nb_files_max = len([entry for entry in os.listdir(self.source_selection_path)])
        print('nbre de fichiers',self.nb_files_max)
        self.num_first_fic = self.extract_number_fic(self.first_image_file)

    def first_image_view(self):
        """
        This method loads the first image in the selected directory with the proper prefix, numbering and format.
        It converts it into a hypermatrix with pixels in row and columns and 4 digits corresponding to the Bayer matrix value
        If the image is in true gray level, this is not a hypermatrix but a "simple" matrix with 1 digit for each pixel location.
        """
        self.preview = mpimg.imread(self.first_image_file)
        plt.clf()
        self.preview_selection = plt.figure(figsize=(15,2))
        
        self.ax = self.preview_selection.subplots()
        self.ax.imshow(self.preview,cmap='binary')
        self.ax.grid(color='black',ls='solid')

        if self.fig_photo_first is not None:
            self.fig_photo_first.get_tk_widget().destroy()
        self.fig_photo_first = FigureCanvasTkAgg(self.preview_selection, master=self.canvas_FOV_ROI)
        self.fig_photo_first.draw()
        self.fig_photo_first.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(self.fig_photo_first, self.canvas_FOV_ROI)
        toolbar.update()
        self.fig_photo_first.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)




    def FEM_select_images_folder(self):
        """
        This method is a dialog box to get the path accessing to the input images
        """

        self.FEM_source_selection_path = filedialog.askdirectory(title='Select the Directory hosting the images')
        self.FEM_nb_files_max = len([entry for entry in os.listdir(self.FEM_source_selection_path)])
        print('nbre de fichiers',self.FEM_nb_files_max)


    def FEM_select_first_image(self):
        """
        This method is a dialog box to get the first input images of the stack of images
        """

        self.FEM_first_image_file = filedialog.askopenfilename(title='Select the first image of the stack of images',initialdir=self.FEM_source_selection_path)
        self.FEM_nb_files_max = len([entry for entry in os.listdir(self.FEM_source_selection_path)])
        print('nbre de fichiers',self.FEM_nb_files_max)
        self.FEM_num_first_fic = self.extract_number_fic(self.FEM_first_image_file)
        print("Numero du premier fichier:",self.FEM_num_first_fic)


    def FEM_first_image_view(self):
        """
        This method loads the first image in the selected directory with the proper prefix, numbering and format.
        It converts it into a hypermatrix with pixels in row and columns and 4 digits corresponding to the Bayer matrix value
        If the image is in true gray level, this is not a hypermatrix but a "simple" matrix with 1 digit for each pixel location.
        """
#        self.preview = mpimg.imread(self.source_selection_path +'/' +self.prefix_entry.get() + str(self.num_images) + str(self.nb_first_image.get()) + self.format_image)
        self.FEM_preview = mpimg.imread(self.FEM_first_image_file)
        plt.clf()
        self.FEM_preview_selection = plt.figure(figsize=(15,2))

        self.FEM_ax = self.FEM_preview_selection.subplots()
        self.FEM_ax.imshow(self.FEM_preview,cmap='binary')
        self.FEM_ax.grid(color='black',ls='solid')

        if self.FEM_fig_photo_first is not None:
            self.FEM_fig_photo_first.get_tk_widget().destroy()
        self.FEM_fig_photo_first = FigureCanvasTkAgg(self.FEM_preview_selection, master=self.FEM_canvas_FOV_ROI)
        self.FEM_fig_photo_first.draw()
        self.FEM_fig_photo_first.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(self.FEM_fig_photo_first,self.FEM_canvas_FOV_ROI)
        toolbar.update()
        self.FEM_fig_photo_first.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)




    def line_select_callback(self,eclick, erelease):
        plt.gcf()
        plt.close('all')

        print('entre dans la selection')

        self.corner_1 = np.round([eclick.xdata, eclick.ydata])
        self.corner_2 = np.round([erelease.xdata, erelease.ydata])
        self.value_coord_corner1.configure(text=str(np.int64(self.corner_1)))
        self.value_coord_corner2.configure(text=str(np.int64(self.corner_2)))
        print('First corner coordinates in px:',self.corner_1)

        print('Second corner coordinates in px:',self.corner_2)

    def select_ROI_rectangle(self):
        plt.close('all')
        plt.gcf()

        rs = RectangleSelector(self.ax, self.line_select_callback,useblit=False, button=[1], 
                            minspanx=5, minspany=5, spancoords='pixels', 
                            interactive=True)

        plt.show()


    def FEM_line_select_callback(self,eclick, erelease):
        plt.gcf()
        plt.close('all')

        print('entre dans la selection')
        self.FEM_corner_1 = np.round([eclick.xdata, eclick.ydata])
        self.FEM_corner_2 = np.round([erelease.xdata, erelease.ydata])
        self.FEM_value_coord_corner1.configure(text=str(np.int64(self.FEM_corner_1)))
        self.FEM_value_coord_corner2.configure(text=str(np.int64(self.FEM_corner_2)))
        print('First FEM corner coordinates in px:',self.FEM_corner_1)

        print('Second FEM corner coordinates in px:',self.FEM_corner_2)

    def FEM_select_ROI_rectangle(self):
        plt.close('all')
        plt.gcf()

        rs = RectangleSelector(self.FEM_ax, self.FEM_line_select_callback,useblit=False, button=[1], 
                            minspanx=5, minspany=5, spancoords='pixels', 
                            interactive=True)

        plt.show()


    def image_stacking(self):
#        self.image_stack = dic.image_stack_from_folder(self.FEM_source_selection_path,file_type=self.FEM_format_image)
        self.image_stack = dic.image_stack_from_folder(self.FEM_source_selection_path,file_type=".tiff")


    def select_menu_num_images(self,event):
        self.num_images = self.list_combo_num_images.get()

    def select_menu_format_image(self,event):
        self.format_image = self.list_combo_format_image.get()

    def FEM_select_menu_num_images(self,event):
        self.num_images = self.list_combo_num_images.get()

    def FEM_select_menu_format_image(self,event):
        self.format_image = self.list_combo_format_image.get()


    def select_menu_FEM_type(self,event):
        self.FEM_type = self.list_combo_FEM_type.get()

    def select_menu_store_internal_var(self,event):
        temp_store_internal_var = self.list_combo_store_internal_var.get()
        if temp_store_internal_var == 'Yes':
            self.temp_store_internals = True
        else:
            self.temp_store_internals = False

    def select_menu_no_convergence_options(self):
        self.no_convergence_action =  self.list_combo_no_convergence_options.get()


    def select_cover_direction(self,event):
        self.cover_direction_value = self.cover_direction[self.list_combo_cover_direction.get()]

    def select_quantity_of_interest_to_plot(self,event):
        self.quantity_of_interest =  self.fields[self.list_combo_quantity_of_interest_to_plot.get()]

    def select_component_quantity_of_interest_to_plot(self,event):
        self.component_quantity_of_interest =  self.component[self.list_combo_component_quantity_of_interest_to_plot.get()]

    def select_text_units(self,event):
        self.text_units =  self.component[self.list_combo_text_units.get()]

    def FEM_generate(self):
        image = Image.open(self.FEM_first_image_file)
        w, h = image.size
        # Caution : the image and the selection of the ROI is carried out based on the np.array
        # The matrix corresponding to the image has its vertical axes inverted compared to the image
        print('height:', h)
       
        xmax = max([np.float64(self.FEM_corner_1)[0],np.float64(self.FEM_corner_2)[0]])
        xmin = min([np.float64(self.FEM_corner_1)[0],np.float64(self.FEM_corner_2)[0]])
        ymax = max([np.float64(self.FEM_corner_1)[1],np.float64(self.FEM_corner_2)[1]])
        ymin = min([np.float64(self.FEM_corner_1)[1],np.float64(self.FEM_corner_2)[1]])
        
        print("xmin =",xmin)
        print("xmax =",xmax)
        print("ymin =",ymin)
        print("ymax =",ymax)

        nb_pix_par_elem = self.nb_pix_per_elem
        nb_elx = int(np.int64(np.floor((xmax-xmin)/nb_pix_par_elem)))
        nb_ely = int(np.int64(np.floor((ymax-ymin)/nb_pix_par_elem)))
        self.mesh = self.mesher.mesh(self.image_stack,Xc1=xmin,Xc2=xmax,Yc1=h-ymax,Yc2=h-ymin,n_ely=nb_ely,n_elx=nb_elx, GUI=False)
        print("Maillage",self.mesh)
        mesh_visu = self.mesher.mesh(self.image_stack,Xc1=xmin,Xc2=xmax,Yc1=ymin,Yc2=ymax,n_ely=nb_ely,n_elx=nb_elx, GUI=False)
        self.FEM_ax.vlines(mesh_visu.xnodes,min(mesh_visu.ynodes),max(mesh_visu.ynodes),color='r')
        self.FEM_ax.hlines(mesh_visu.ynodes,min(mesh_visu.xnodes),max(mesh_visu.xnodes),color='r')

        self.FEM_fig_photo_first.get_tk_widget().destroy()

        self.FEM_fig_photo_first = FigureCanvasTkAgg(self.FEM_preview_selection, master=self.FEM_canvas_FOV_ROI)
        self.FEM_fig_photo_first.draw()
        self.FEM_fig_photo_first.get_tk_widget().pack()
        # toolbar = NavigationToolbar2Tk(self.FEM_fig_photo_first,self.FEM_canvas_FOV_ROI)
        # toolbar.update()
        # self.FEM_fig_photo_first.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


    def extract_number_fic(self,nom_fic=str):
        """
        The purpose of this method is to get the digits of the raw images.
        We assume that the numbering of the files is at the end of name of the file just before the extension.
        Arg : Name of file with or without its path
        Returns : the number of the current selected file.
        """
        nom_fic = Path(nom_fic).stem
        number=''
        for c in reversed(nom_fic):
            if c.isdigit(): 
                number = number+c
            else:
                break
        return number[::-1]


    def crop_images_to_ROI(self):
        self.nb_images_to_crop = np.int64(self.prepare_resize_ROI_files_entry.get())
        self.num_images = self.list_combo_num_images.get()
        for i in range(1,self.nb_images_to_crop+1):
            if len(self.num_images)==1:
                num_image ="{:02n}".format(i)
            elif len(self.num_images)==2:
                num_image ="{:03n}".format(i)
            elif len(self.num_images)==3:
                num_image ="{:04n}".format(i)
            elif len(self.num_images)==4:
                num_image ="{:05n}".format(i)
            elif len(self.num_images)==5:
                num_image ="{:06n}".format(i)
            elif len(self.num_images)==6:
                num_image ="{:07n}".format(i)

            #num_image ="{:0"+str(self.num_images+1)+"n}".format(i)
            #nom_image = "'"+repertoire+nom_fic_base+num_image+'.tiff'+"'"
            #print(nom_image)
            image_entree = (self.first_image_file)
            
            image = Image.open(image_entree) # On acquiert l'image
            image_input = np.asarray(image)

            x_min = np.int64(np.round(min(self.corner_1[1],self.corner_2[1])))
            x_max = np.int64(np.round(max(self.corner_1[1],self.corner_2[1])))
            y_min = np.int64(np.round(min(self.corner_1[0],self.corner_2[0])))
            y_max = np.int64(np.round(max(self.corner_1[0],self.corner_2[0])))




            image_cut_on_ROI = image_input[x_min:x_max,y_min:y_max] 

            Image.fromarray(image_cut_on_ROI).save(self.output_path +'/' +self.prefix_ROI_files_entry.get() +num_image + self.format_image)
            if i == (self.nb_images_to_crop):
                print('Images have all been cropped !')

    def create_output_folder(self):
        output_dir = self.output_ROI_dir_entry.get()
        self.output_path = os.path.join(self.source_selection_path +'/' + output_dir) # dir is a directory taken from the filedialog
        if os.path.exists(self.output_path) == True: #Only creates a new folder when non-existing
            pass
        else:
            os.mkdir(self.output_path)
        print(self.output_path)

    def FEM_mesh_prop_gen(self):
        self.mesher = dic.Mesher(deg_e=2, deg_n=2,type=self.list_FEM_type)

    def DIC_generate_settings(self):
        self.DIC_settings = dic.DICInput(self.mesh,self.image_stack)
        #self.DIC_settings.ref_update = tuple(self.first_image_for_DIC_entry.get())
        self.DIC_settings.max_nr_im = int(np.int64(self.FEM_nb_stack_images.get()))
        self.DIC_settings.maxit = int(np.int64(self.iteration_entry.get()))
        self.DIC_settings.tom = np.float64(self.convergence.get())
        self.DIC_settings.interpolation_order = int(np.int64(self.order_interp.get()))
        self.DIC_settings.store_internals=self.temp_store_internals
        self.DIC_settings.noconvergence = "ignore"





    def prepare_DIC_analysis(self):
        self.DIC_job = dic.DICAnalysis(self.DIC_settings)

    def run_DIC_analysis(self):
        self.DIC_results = self.DIC_job.run()
        print('DIC finished ! '+self.FEM_nb_stack_images.get()+' images have been processed.')

    def generate_animated_gif(self,images_list: list,quantity_of_interest: str,analysis: str,output_directory: Path,duration_im: float):
        '''
        This method generates an animated gif to visualize the evolution of a quantity of interest along time
        Args :  images_list: A list with the different images of processing considering a specific quantity of interest
                quantity_of_interest: The quantity of interest figured out in the input images
                analysis: The name of the analysis that is carried out and that will be the suffix of the ouput file name
                output_directory: The path of the directory where the output animated .gif will be written
                duration_im: Duration of appareance of each individual image 
        Returns : An animated .gif
        '''
        # frames = []
        # for i in image_set:
        #     new_frame = Image.open(i)
        #     frames.append(new_frame)
        images_list[0].save(output_directory + analysis + quantity_of_interest + '.gif', format='GIF',
              append_images=images_list[1:],
              save_all=True,
              duration=duration_im, loop=0)
        return images_list


    def calculate_fields(self):
        """
        This method calculates all the fields of quantities of interest at every instant and for every elements
        Arg : The result of the DIC
        Returns : An object containing the different fields that can be extracted with different methods self.DIC_results
        """
        self.all_fields = dic.post.viz.Fields(self.DIC_results,upscale=1)

    def extract_fields_per_frame(self,considered_frame):
        """
        This method extracts the quantity of interest and its component for every finite element at every instant
        The coordinates in n_y direction are changed to cope with the orientation of the photoreceptor of the lense.
        The coordinate system of a camera is a non direct coordinate system y is downward and x from the left to the right
        Args : the object containing every fields, the considered instant
        Returns :   3 matrices:
                    Matrix with quantity of interest of size n_x x n_y
                    Matrix with the coordinates along n_x
                    Matrix with the coordinates along n_y            
        """
        method_for_the_selected_field = self.fields[self.list_combo_quantity_of_interest_to_plot.get()]
        component = self.component[self.list_combo_component_quantity_of_interest_to_plot.get()]
        quantity = self.all_fields.method_for_the_selected_field
        if method_for_the_selected_field == 'disp()':
            self.element_coord_in_x = quantity[0,:,:,:,considered_frame][0,0,:]
            self.element_coord_in_y = quantity[0,:,:,:,considered_frame][0,:,0]
            if component == 'Disp-x':
                self.field_to_plot_any_time = quantity[0,:,:,:,considered_frame][0,:,:][0]
            else:
                self.field_to_plot_any_time = quantity[0,:,:,:,considered_frame][0,:,:][1]
        else:
            self.element_coord_in_x = quantity[0,:,:,:,:,considered_frame][0,0,0,:]
            self.element_coord_in_y = quantity[0,:,:,:,:,considered_frame][0,0,:,0]
            if component == 'xx':
                self.field_to_plot_any_time = quantity[0,:,:,:,:,considered_frame][0,:,:,:][0]
            elif component == 'yy':
                self.field_to_plot_any_time = quantity[0,:,:,:,:,considered_frame][:,0,:,:][0]
            else: # for the component 'xy'
                self.field_to_plot_any_time = quantity[0,:,:,:,:,considered_frame][:,0,:,:][1]
                # Because of Cauchy principle for isotropic medium we could have chosen also [0,:,:,:][1]
                # as epsilon_xy = epsilon_yx 

    def extract_fields_for_frame_list(self,frame_list=list):
        self.selected_field_for_frame_list=[]
        for i in frame_list:
            self.selected_field_for_frame_list.append(self.extract_fields_per_frame(i))

    def plot_fields(self,selected_frame=int):
        print('toto')

    def select_ref_length(self):
        print('ref_length')


    def cal_residual(self,disp_pixel_one_frame):
        """
        This method calculates the residual value per finite element
        """
        print("calc_residual")

    def plot_fringe(self):
        print('plot fringe')

    def compute_test_char(self):
        print('Compute test characteristics')
        self.H_FOV = np.float64(self.distance_to_target_value.get())*np.float64(self.sensor_size_H_value.get())/np.float64(self.focal_length_value.get())
        self.V_FOV = np.float64(self.distance_to_target_value.get())*np.float64(self.sensor_size_V_value.get())/np.float64(self.focal_length_value.get())
        self.H_FOV_value.configure(text=str(round(np.float64(self.H_FOV),3))+' m')
        self.V_FOV_value.configure(text=str(round(np.float64(self.V_FOV),3))+' m')

        # self.resolution_H_FOV = 1./(np.float64(self.pixel_numer_H_cam_value.get())/self.H_FOV/1000.)
        self.resolution_FOV = 1./(np.float64(self.pixel_numer_V_cam_value.get())/self.V_FOV/1000.)
        # print(self.resolution_H_FOV)
        # print(self.resolution_V_FOV)
        # self.definition_FOV_value.configure(text=str(round(np.float64(self.H_FOV),3))+' m')
        self.resolution_FOV_value.configure(text=str(round(1./(np.float64(self.pixel_numer_V_cam_value.get())/self.V_FOV/1000.),3))+' mm/px')
        self.definition_FOV_value.configure(text=str(round(1./(np.float64(self.pixel_numer_V_cam_value.get())/self.V_FOV/1000.)*np.float64(self.smallest_size_text_value.get()),3)))
        def_FOV = (np.float64(self.pixel_numer_V_cam_value.get())/self.V_FOV/1000.)
        self.speckle_diam_value.configure(text=str(round(np.float64(self.nb_pixel_per_spot_value.get())/np.float64(def_FOV),1)))
        # sens_size =  
        # self_H_
        self.angle_of_view_H = round(2*np.arctan(np.float64(self.sensor_size_H_value.get())/2/np.float64(self.focal_length_value.get()))*180/np.pi,1)
        self.angle_of_view_V = round(2*np.arctan(np.float64(self.sensor_size_V_value.get())/2/np.float64(self.focal_length_value.get()))*180/np.pi,1)
        #print(self.angle_of_view_H)

        self.angle_of_view_V_value.configure(text=str(self.angle_of_view_V))
        self.angle_of_view_H_value.configure(text=str(self.angle_of_view_H))

        self.cover_direction_value = self.cover_direction[self.list_combo_cover_direction.get()]
        self.overlap_length_V = round((np.float64(self.cover_pct_value.get())/100.)*self.V_FOV,2)
        self.overlap_length_V_value.configure(text=str(self.overlap_length_V))
        self.overlap_length_H = round((np.float64(self.cover_pct_value.get())/100.)*self.H_FOV,2)
        self.overlap_length_H_value.configure(text=str(self.overlap_length_H))
        
        if self.cover_direction_value == 'Vertical':
            self.distance_2_cams = round((1.-np.float64(self.cover_pct_value.get())/100.)*self.V_FOV,2)
            self.distance_2_cams_value.configure(text=str(self.distance_2_cams))
            #print(self.distance_2_cams)
            # self.setup_drawing(self.number_of_camera_text_value,self.distance_2_cams_value,self.distance_to_target_value,self.V_FOV_value,self.cover_pct_value,self.overlap_length_V_value)
            #print(str(self.H_FOV-np.float64((self.cover_pct_value.get()))/100.*self.H_FOV))
            # self.sketch_characteristics = self.test_ketch_characteristics(number_of_camera_text_value= np.int64(self.number_of_camera_text_value.get()),distance_2_cams_value=self.distance_2_cams,distance_to_target_value=np.float64(self.distance_to_target_value.get()),FOV_value=self.H_FOV,cover_pct_value=np.float64(self.cover_pct_value.get()),overlap_length_value=self.overlap_length_H)
            self.plot_sketch()
        elif self.cover_direction_value == 'Horizontal':
            self.distance_2_cams = round((1.-np.float64(self.cover_pct_value.get())/100.)*self.H_FOV,2)
            #print(self.distance_2_cams)
            self.distance_2_cams_value.configure(text=str(self.distance_2_cams))
            # print(str(self.H_FOV))
            # print(str(np.float64((self.cover_pct_value.get()))/100.*self.H_FOV))
            # print(str(np.float64(self.cover_pct_value.get())/100.))
            # print(str(self.H_FOV-np.float64((1.-np.float64(self.cover_pct_value.get())/100.)*self.H_FOV)))
            # print(np.float64((self.cover_pct_value.get()))/100.*self.H_FOV-np.float64((1.-np.float64(self.cover_pct_value.get())/100.)*self.H_FOV))
            #print(str(self.H_FOV-np.float64((self.cover_pct_value.get()))/100.*self.H_FOV))
        self.plot_sketch()


    def plot_sketch(self):
        """
        This function generates a plot that figures out the top and side view of the experimental camera(s) setup
        Args:   number_of_camera = Integer # This figures out if one or two can are used.
                distance_2_cams_value = float # This figures out the distance between two cameras
                distance_to_target_value = float # This figures out the distance between the camera lens and the focal plan
                FOV_value = float # This figures out the field of view of a single camera 
                cover_pct_value = 10. # Value in % of overlapping of the field of view of the two cameras
        Returns:
                sketch_characteristics = Dictionnary with 3 keys overlap, camera_1 and camera_2
                * overlap key refers to a float that corresponds to the overlapping length of the txo cameras
                * camera_2 keys refers to a dictionnary with 3 keys:
                    - cam_pos = Camera position
                    - FOV_min_pos = the location of the minimum value of the field of view of the considered camera
                    - FOV_max_pos = the location of the maximum value of the field of view of the considered camera
        """
        if self.canvas_fig_setup_sketch is not None:
            self.canvas_fig_setup_sketch.destroy()
        self.H_FOV = np.float64(self.distance_to_target_value.get())*np.float64(self.sensor_size_H_value.get())/np.float64(self.focal_length_value.get())
        self.V_FOV = np.float64(self.distance_to_target_value.get())*np.float64(self.sensor_size_V_value.get())/np.float64(self.focal_length_value.get())
        distance_2_cams_value = self.distance_2_cams
        #distance_2_cams_value = (1.-np.float64(self.cover_pct_value.get())/100.)*self.H_FOV
        distance_to_target_value = np.float64(self.distance_to_target_value.get())
        if self.cover_direction[self.list_combo_cover_direction.get()] == 'Horizontal':
            FOV_value = self.H_FOV
            overlap_length_value = self.overlap_length_H
        else:
            FOV_value = self.V_FOV
            overlap_length_value = self.overlap_length_V
        self.sketch_characteristics = {'camera_1': {'cam_pos': [0,0],
                                        'FOV_min_pos':[distance_to_target_value,-FOV_value/2-np.float64(self.cover_pct_value.get())/100*FOV_value/2],
                                        'FOV_max_pos':[distance_to_target_value,FOV_value/2+np.float64(self.cover_pct_value.get())/100*FOV_value/2]},
                        'camera_2': {'cam_pos': [0,distance_2_cams_value],
                        'FOV_min_pos':[distance_to_target_value,-FOV_value/2+distance_2_cams_value-np.float64(self.cover_pct_value.get())/100*FOV_value/2],
                        'FOV_max_pos':[distance_to_target_value,FOV_value/2+distance_2_cams_value+np.float64(self.cover_pct_value.get())/100*FOV_value/2]},
                        'overlap': np.float64(self.cover_pct_value.get())*FOV_value/100,
                        'cam_nbr':np.int64(self.number_of_camera_text_value.get())}

        # self.calc_test_ketch_characteristics()
        #print(self.sketch_characteristics)
        #number_of_camera_text_value = np.float64(self.number_of_camera_text_value.get())

        #print(self.sketch_characteristics['cam_nbr'])

        if self.sketch_characteristics['cam_nbr']==1:


            self.ax_sketch  = plt.figure()
            self.ax_sketch.clear()
            self.ax_sketch.clf()
            # self.self.self.ax_sketch = plt.subplots(111)
            # self.self.self.ax_sketch.get_autoscale_on()
            fig_sketch = plt.figure(figsize=(21,72))

            self.ax_sketch = fig_sketch.add_subplot(111)
            # Plot the dimension between the two cams
            self.ax_sketch.annotate(text='', xy=(self.sketch_characteristics['camera_1']['cam_pos'][0],
                                    self.sketch_characteristics['camera_1']['cam_pos'][1]),
                        xytext=(
                            self.sketch_characteristics['camera_2']['cam_pos'][0],
                                self.sketch_characteristics['camera_2']['cam_pos'][1]),
                            arrowprops=dict(arrowstyle='<->',
                                            shrinkA=0, shrinkB=0))
            middle_segment_dist_2_cams = (self.sketch_characteristics['camera_2']['cam_pos'][1]-self.sketch_characteristics['camera_1']['cam_pos'][1])/2
            self.ax_sketch.text(x=self.sketch_characteristics['camera_1']['cam_pos'][0],y=middle_segment_dist_2_cams,s=str(np.round(distance_2_cams_value,2))+' m',rotation=90,rotation_mode = 'anchor',transform_rotates_text=True)
            # Plot the dimension between the cams and the focal plan
            self.ax_sketch.annotate(text='', xy=(self.sketch_characteristics['camera_2']['cam_pos'][0],
                                    self.sketch_characteristics['camera_2']['cam_pos'][1]/2),
                        xytext=(
                            self.sketch_characteristics['camera_2']['cam_pos'][0]+distance_to_target_value,
                                self.sketch_characteristics['camera_2']['cam_pos'][1]/2),
                            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
            middle_segment_dist_to_target = (distance_to_target_value/2)
            self.ax_sketch.text(x=middle_segment_dist_to_target,y=(self.sketch_characteristics['camera_2']['cam_pos'][1]-self.sketch_characteristics['camera_1']['cam_pos'][1])/2,s=str(distance_to_target_value)+' m',rotation=0,rotation_mode = 'anchor',transform_rotates_text=True)
            
            # Plot the overlapping dimension
            self.ax_sketch.annotate(text='', xy=(self.sketch_characteristics['camera_1']['FOV_max_pos'][0],
                                    self.sketch_characteristics['camera_1']['FOV_max_pos'][1]),
                        xytext=(
                            self.sketch_characteristics['camera_2']['FOV_min_pos'][0],
                                self.sketch_characteristics['camera_2']['FOV_min_pos'][1]),
                            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
            # Plot the FOV length
            self.ax_sketch.annotate('',xy=(self.sketch_characteristics['camera_1']['FOV_max_pos'][0],
                                self.sketch_characteristics['camera_1']['FOV_max_pos'][1]),
                        xytext=(self.sketch_characteristics['camera_1']['FOV_min_pos'][0],
                                self.sketch_characteristics['camera_1']['FOV_min_pos'][1]),
                            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
            middle_segment_FOV = (self.sketch_characteristics['camera_1']['FOV_min_pos'][1])/2
            self.ax_sketch.text(x=distance_to_target_value,y=middle_segment_FOV,s=str(round(FOV_value,2))+' m',rotation=90,rotation_mode = 'anchor',transform_rotates_text=True)

            x_values_seg_cam1_FOV_min = [self.sketch_characteristics['camera_1']['cam_pos'][0],self.sketch_characteristics['camera_1']['FOV_min_pos'][0]]
            y_values_seg_cam1_FOV_min = [self.sketch_characteristics['camera_1']['cam_pos'][1],self.sketch_characteristics['camera_1']['FOV_min_pos'][1]]

            x_values_seg_cam1_FOV_max = [self.sketch_characteristics['camera_1']['cam_pos'][0],self.sketch_characteristics['camera_1']['FOV_max_pos'][0]]
            y_values_seg_cam1_FOV_max = [self.sketch_characteristics['camera_1']['cam_pos'][1],self.sketch_characteristics['camera_1']['FOV_max_pos'][1]]


            x_values_seg_cam2_FOV_min = [self.sketch_characteristics['camera_2']['cam_pos'][0],self.sketch_characteristics['camera_2']['FOV_min_pos'][0]]
            y_values_seg_cam2_FOV_min = [self.sketch_characteristics['camera_2']['cam_pos'][1],self.sketch_characteristics['camera_2']['FOV_min_pos'][1]]

            x_values_seg_cam2_FOV_max = [self.sketch_characteristics['camera_2']['cam_pos'][0],self.sketch_characteristics['camera_2']['FOV_max_pos'][0]]
            y_values_seg_cam2_FOV_max = [self.sketch_characteristics['camera_2']['cam_pos'][1],self.sketch_characteristics['camera_2']['FOV_max_pos'][1]]


            self.ax_sketch.plot(x_values_seg_cam1_FOV_min,y_values_seg_cam1_FOV_min,'bo',linestyle='--')
            self.ax_sketch.plot(x_values_seg_cam1_FOV_max,y_values_seg_cam1_FOV_max,'bo',linestyle='--')
            self.ax_sketch.plot(x_values_seg_cam2_FOV_min,y_values_seg_cam2_FOV_min,'bo',linestyle='--')
            self.ax_sketch.plot(x_values_seg_cam2_FOV_max,y_values_seg_cam2_FOV_max,'bo',linestyle='--')

            middle_segment_overlap= self.sketch_characteristics['camera_1']['FOV_max_pos'][1]
            self.ax_sketch.text(x=distance_to_target_value,y=middle_segment_overlap,s='overlap '+str(overlap_length_value)+' m',rotation=0,rotation_mode = 'anchor',transform_rotates_text=True)


            self.ax_sketch.set_xlim(-0.25, distance_to_target_value+0.25)
            self.ax_sketch.set_ylim(self.sketch_characteristics['camera_1']['FOV_min_pos'][1]-0.25, self.sketch_characteristics['camera_2']['FOV_max_pos'][1]+0.25)
            self.ax_sketch.text(x=self.sketch_characteristics['camera_1']['cam_pos'][0]-0.25,y=self.sketch_characteristics['camera_1']['cam_pos'][1]-0.25,s='Camera #1')

            self.ax_sketch.text(x=self.sketch_characteristics['camera_2']['cam_pos'][0]-0.25,y=self.sketch_characteristics['camera_2']['cam_pos'][1]+0.25,s='Camera #2')
            self.ax_sketch.set_aspect('equal', 'box')


            self.canvas_fig_setup_sketch = tk.Canvas(self.experimental_sketch)
    #        self.canvas_FOV_ROI.pack(expand=1, fill='both')
            
            self.canvas_fig_setup_sketch.pack(expand=1,fill='both')
            self.Figurecanvas_fig_setup_sketch = FigureCanvasTkAgg(figure=fig_sketch, master=self.canvas_fig_setup_sketch)

            self.Figurecanvas_fig_setup_sketch.draw()
            self.Figurecanvas_fig_setup_sketch.get_tk_widget().pack()
            # toolbar = NavigationToolbar2Tk(self.Figurecanvas_fig_setup_sketch,)
            # toolbar.update().pack(fill='both',expand=True)
            # self.Figurecanvas_fig_setup_sketch.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            
            # self.fig_photo_first = FigureCanvasTkAgg(self.preview_selection, master=self.canvas_FOV_ROI)
            # self.fig_photo_first.draw()
            # self.fig_photo_first.get_tk_widget().pack()
            # toolbar = NavigationToolbar2Tk(self.fig_photo_first, self.canvas_FOV_ROI)
            # toolbar.update()
            # self.fig_photo_first.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)






    def create_gui(self):
        """
        Creates the GUI

        This method uses the tkinter class to generate the different graphical elements of the GUI: buttons, text areas and tabs.
        It calls the different methods of the muDIC_GUI class
        """
        # Pre-processing of images
        tab_control = ttk.Notebook(self.root)

        self.text_icone = u"\N{GREEK SMALL LETTER MU}"+'DIC'+u"\u1D33"+u"\u1D41"+u"\u1D35"
        self.text_muDIC = u"\N{GREEK SMALL LETTER MU}"+'DIC'

        tab_info = ttk.Frame(tab_control)
        tab_control.add(tab_info, text='How to use '+self.text_icone+' / Quit the application')
        tab_control.pack(expand=1
                         , fill='both'
                         )





        tab0 = ttk.Frame(tab_control)
        tab_control.add(tab0, text='Test campaign preparation')
        tab_control.pack(expand=1
                         , fill='both'
                         )
        tab0.grid_columnconfigure(0, weight=1)
        tab0.grid_rowconfigure(1,weight=1)

        tab1 = ttk.Frame(tab_control)
        tab_control.add(tab1, text='Pre-processing')
        tab_control.pack(expand=1
                         , fill='both'
                         )
        tab1.grid_columnconfigure(0, weight=1)
        tab1.grid_rowconfigure(1,weight=1)
        # tab1.grid_size()

        # Definition of the ROI, creation of the mesh and calculation
        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text='Digital Image Correlation')
        tab_control.pack(expand=1
                         , fill='both'
                         )
        tab2.grid_columnconfigure(0, weight=1)
        tab2.grid_rowconfigure(1,weight=1)

        # Post-processing of the different quantities of interest
        tab3 = ttk.Frame(tab_control)
        tab_control.add(tab3, text='Post-processing fields')
        tab_control.pack(expand=1
                         , fill='both'
                         )
        tab3.grid_columnconfigure(0, weight=1)
        tab3.grid_rowconfigure(1,weight=1)

        tab3bis = ttk.Frame(tab_control)
        tab_control.add(tab3bis, text='Post-processing local data')
        tab_control.pack(expand=1
                         , fill='both'
                         )
        tab3bis.grid_columnconfigure(0, weight=1)
        tab3bis.grid_rowconfigure(1,weight=1)




        tab4 = ttk.Frame(tab_control)
        tab_control.add(tab4, text='DIC - Code_Aster dialogue')
        tab_control.pack(expand=1
                         , fill='both'
                         )
        tab4.grid_columnconfigure(0, weight=1)
        tab4.grid_rowconfigure(1,weight=1)

        tab5 = ttk.Frame(tab_control)
        tab_control.add(tab5, text='About')
        tab_control.pack(expand=1
                         , fill='both'
                         )
        tab5.grid_columnconfigure(0, weight=1)
        tab5.grid_rowconfigure(1,weight=1)


#####################################################################################
        ############################################################################
        # Code lines related to presentation of muDIC-GUI and Quit the application #
        ############################################################################
#####################################################################################

        font_size_text_description = 9
        frame_width_description = 200
        style_font_description = ('Arial', font_size_text_description)

#########################################################################################
        # Description of the different tabs of the application
        ##########################################################
        frame_description_test_prep = ttk.LabelFrame(tab_info,text='Test campaign preparation')
        frame_description_test_prep.pack(
             expand=1,
              fill='both',
             # side='left',
            #   padx=2, pady=2
              )
        self.FRAME_for_icone=ttk.Frame(frame_description_test_prep, width=10, height =10)
        self.FRAME_for_icone.place(in_=frame_description_test_prep, relx=1.0, x=-150, rely=0, y=5)

        LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()




        description_test_prep = 'The purpose of this tab is to set up the best achievable parameters in order to capture digital images on a test sample. The user will need to know \n   (1)The characteristics of the camera (sensor size, pixel size per unit length, lens, aperture. \n   (2)the largest size of the sample that will be observed, the magnitude of the quantities of interest that are expected to get. \nThis part will provide the size of the particles of the speckle to reach the expected magnitude of the quantities of interest.'

        T1 = ttk.Label(frame_description_test_prep
                      ,text=description_test_prep
                      ,justify='left'
                      ,width=frame_width_description
                      ,font=style_font_description
                      )
        T1.pack(expand=1,anchor='nw')


        frame_description_preproc = ttk.LabelFrame(tab_info,text='Pre-processing')
        frame_description_preproc.pack(
             expand=1,
              fill='both',
             # side='left',
            #   padx=2, pady=2
              )

        description_preproc = 'The purpose of this tab is to prepare the images that have been captured during the test campaign. \nThe user shall have prepared the data so that they are as follows:\n   (1) Raw images - This means that they shall not have been compressed. TIF, TIFF, PNG, etc .. will be prefered. \n   (2) The image files shall terminate by their numbering: for example a set of name_of_the_file00xxx.tif. Where xxx can start at any value. The amount of 0 before the number can vary. \n   (3)The files shall be in the same directory. \n The user can resize the images in order to get rid of elements in the field of view that are not useful in the analysis. \n The user will specify different options to store the cropped images as: \n   (1)The prefix of the output images. \n   (2)The target directory where they will be stored. \n   (3)The format of the numbering of the images. \n   (4)The extension of the files: .tiff, .png, etc ...'

        T1 = ttk.Label(frame_description_preproc
                      ,text=description_preproc
                      ,justify='left'
                      ,width=frame_width_description
                      ,font=style_font_description
                      )
        T1.pack(expand=1,anchor='nw')





        frame_DIC = ttk.LabelFrame(tab_info,text='Digital Image Correlation')
        frame_DIC.pack(
             expand=1,
              fill='both',
             # side='left',
            #   padx=2, pady=2
              )


        description_DIC = 'The purpose of this tab is to prepare and run the DIC analysis. \nFirst the user will identify the ROI on the image by selecting it on the preview of the first image of the set of captured data. \nThen it possible to chose the properties of the mesh that is generated: \n   (1) The type of element: Q4 or T3 (not supported for the moment) \n   (2) The amount of pixels per element side. The element are built as square as possible considering the size of the ROI and the density of pixels per element. \nWhen the ROI is meshed, it is possible to run the muDIC solver, by selecting it different settings as:\n   (1)The reference image \n   (2)The last image that is analyzed\n   (3)The max. number of iterations of each convergence step\n   (4)The tolerance for each convergence step\n   (5)...'

        T1 = ttk.Label(frame_DIC,
                      text=description_DIC,
                      justify='left',
                      width=frame_width_description,
                      font=style_font_description
                      )
        T1.pack(expand=1,anchor='nw')



        frame_description_postproc = ttk.LabelFrame(tab_info,text='Post-processing')
        frame_description_postproc.pack(
             expand=1,
              fill='both',
             # side='left',
            #   padx=2, pady=2
              )


        description_postproc = 'The purpose of this tab is to post-process the result of the DIC solver.\nFor this purpose, the user will have the choice to keep the pixels as unit or to define with a GUI on clicking on a reference on screen whose actual length is well known.\nThe scale parameter between pixels and actual unit that have been selected by the user (mm,cm,m)\nIt is possible to select amoung the different quantities of interest produced by the solver and that the user can scroll.\nIt is also possible to pick a specific point and plot its evolution (displacement, velocity, acceleration) along time.\nThen it is possible to calculate SDOF-Response spectrum or to carry out Cauchy Continuous Wavelet Transform analysis.'

        T1 = ttk.Label(frame_description_postproc,
                      text=description_postproc,
                      justify='left',
                      width=frame_width_description,
                      font=style_font_description
                      )
        T1.pack(expand=1,anchor='nw')






        frame_description_code_aster_dialogue = ttk.LabelFrame(tab_info,text='DIC-Code_Aster dialogue')
        frame_description_code_aster_dialogue.pack(
             expand=1,
              fill='both',
             # side='left',
            #   padx=2, pady=2
              )

        description_DIC_Code_Aster = 'This last tab provides the possibility to prepare a Code_Aster computation based on the meshed ROI and its boundary conditions in displacement along time.\nA Code_Aster command file is generated. It accounts for:\n   (1)A condensation of the displacement on a polynomial basis, along time and along the segment of the sample that are selected on screen by the user.\n   (2)A set of material properties among those that are proposed in Code_Aster (focusing on elastic, viscoplastic or damage constitutive equations.'
        T1 = ttk.Label(frame_description_code_aster_dialogue,
                      text=description_DIC_Code_Aster,
                      justify='left',
                      width=frame_width_description,
                      font=style_font_description
                      )
        T1.pack(expand=1,anchor='nw')


        frame_for_button_and_icone = ttk.LabelFrame(tab_info,text='Quit the application')
        frame_for_button_and_icone.pack(
             expand=1,
              fill='y',
             # side='left',
            #   padx=2, pady=2
              )





        FRAME_for_quit=ttk.Frame(frame_for_button_and_icone,
                                # width=10,
                                # height =10
                                )
        FRAME_for_quit.pack()

        # FRAME_for_quit.place(in_=frame_for_button_and_icone, relx=1.0, x=-250)

        myFont_quit = font.Font(weight="normal")
        quit_button = tk.Button(FRAME_for_quit,text = "Quit", fg='Red', command = self.Close
                                # ,width=5
                                )
        # "Run "+u"\N{GREEK SMALL LETTER MU}"+"DIC"
        # Define font
        quit_button['font'] = myFont_quit
        quit_button.grid(row=1, column=5,
                        #  padx=2,
                        #  pady=2
                         )
        #quit_button.place(in_=preprocessing_frame,relx=1.0, x=-130, rely=0, y=50)


#####################################################################################
        #######################################################################
        # Code lines related to the experimental preparation / speckle choice #
        #######################################################################
#####################################################################################
        # test_prep_frame = ttk.LabelFrame(tab0, text='Test campaign preparation')
        # test_prep_frame.pack(expand=1, fill='x', padx=2, pady=2)


        # self.FRAME_for_icone=ttk.Frame(test_prep_frame, width=10, height =10)
        # self.FRAME_for_icone.place(in_=test_prep_frame, relx=1.0, x=-150, rely=0, y=5)

        # LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()

        # Frame pour user data
        setup_properties = ttk.LabelFrame(tab0, text='Setup properties')
        # preprocessing_frame.pack(expand=1,fill='x',padx=2, pady=2)
        
        self.experimental_sketch = ttk.LabelFrame(tab0, text='Sketch of exprimental setup')
        
        # self.preview_first_image_frame.pack(expand=0, fill='x', padx=2, pady=2)

        experimental_capabilities = ttk.LabelFrame(tab0, text='Experimental capabilitie and speckle recommandation',height=100)
        # ROI_frame.pack(expand=1, fill='x', padx=2, pady=2)

        # setup_properties.grid(row=0, column=0, sticky="nswe")
        # self.experimental_sketch.grid(row=1, column=0, sticky="nswe",pady=2)
        # experimental_capabilities.grid(row=2, column=0, sticky="nswe")
        setup_properties.grid(row=0, column=0, sticky="nswe")
        self.experimental_sketch.grid(row=1, column=0, sticky="nswe",pady=2,rowspan=2)
        experimental_capabilities.grid(row=2, column=0, sticky="nswe")

############################################################################
        # Text area to define the prefix of the images
        focal_length_text = 'Focal length [mm]:'
        focal_length = ttk.Label(setup_properties, text=focal_length_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(focal_length_text))
        focal_length.grid(row=1, column=6, padx=2, pady=2)
        self.focal_length_value = ttk.Entry(setup_properties,width=5)
        self.focal_length_value.insert(0, 28)
        self.focal_length_value.grid(row=1, column=7, padx=2, pady=2)

        sensor_size_H_text = 'Sensor size (H)[mm]:'
        sensor_size_H = ttk.Label(setup_properties, text=sensor_size_H_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(sensor_size_H_text))
        sensor_size_H.grid(row=0, column=2, padx=2, pady=2)
        self.sensor_size_H_value = ttk.Entry(setup_properties,width=5)
        self.sensor_size_H_value.insert(0, 16.35)
        self.sensor_size_H_value.grid(row=0, column=3, padx=2, pady=2)

        sensor_size_V_text = '(V)[mm]:'
        sensor_size_V = ttk.Label(setup_properties, text=sensor_size_V_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(sensor_size_V_text))
        sensor_size_V.grid(row=0, column=4, padx=2, pady=2)
        self.sensor_size_V_value = ttk.Entry(setup_properties,width=5)
        self.sensor_size_V_value.insert(0, 12.1)
        self.sensor_size_V_value.grid(row=0, column=5, padx=2, pady=2)

        self.pixel_input= tk.IntVar(value=1)

        ttk.Radiobutton(setup_properties,text='Pixel size',variable=self.pixel_input, value=1).grid(row=1, column=0, padx=2, pady=2)
        ttk.Radiobutton(setup_properties,text='Number of pixels',variable=self.pixel_input, value=2).grid(row=1, column=2, padx=2, pady=2)

        pixel_size_cam_text = 'Px size in '+u"\N{GREEK SMALL LETTER MU}"+'m:'
        pixel_size_cam = ttk.Label(setup_properties, text=pixel_size_cam_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(pixel_size_cam_text))
        pixel_size_cam.grid(row=2, column=0, padx=2, pady=2)
        self.pixel_size_cam_value = ttk.Entry(setup_properties,width=5)
        self.pixel_size_cam_value.insert(0, 3.76)
        self.pixel_size_cam_value.grid(row=2, column=1, padx=2, pady=2)

        pixel_numer_H_cam_text = 'Number of pixels (H):'
        pixel_numer_H_cam = ttk.Label(setup_properties, text=pixel_numer_H_cam_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(pixel_numer_H_cam_text))
        pixel_numer_H_cam.grid(row=2, column=2, padx=2, pady=2)
        self.pixel_numer_H_cam_value = ttk.Entry(setup_properties,width=5)
        self.pixel_numer_H_cam_value.insert(0, 2336)
        self.pixel_numer_H_cam_value.grid(row=2, column=3, padx=2, pady=2)

        pixel_numer_V_cam_text = '(V) [mm]:'
        pixel_numer_V_cam = ttk.Label(setup_properties, text=pixel_numer_V_cam_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(pixel_numer_V_cam_text))
        pixel_numer_V_cam.grid(row=2, column=4, padx=2, pady=2)
        self.pixel_numer_V_cam_value = ttk.Entry(setup_properties,width=5)
        self.pixel_numer_V_cam_value.insert(0, 1728)
        self.pixel_numer_V_cam_value.grid(row=2, column=5, padx=2, pady=2)


        # Text area to define the prefix of the images
        distance_to_target_text = 'Distance to target [m]:'
        distance_to_target = ttk.Label(setup_properties, text=distance_to_target_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(distance_to_target_text))
        distance_to_target.grid(row=0, column=6, padx=2, pady=2)
        self.distance_to_target_value = ttk.Entry(setup_properties,width=5)
        self.distance_to_target_value.insert(0, 2.5)
        self.distance_to_target_value.grid(row=0, column=7, padx=2, pady=2)

        self.camera_type_dict = {}
        self.camera_type_dict['Mono']='Mono'
        self.camera_type_dict['Color']='Color'


        camera_type_text = 'Camera type:'
        camera_type = ttk.Label(setup_properties, text=camera_type_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(camera_type_text))
        camera_type.grid(row=0, column=0, padx=2, pady=2)
       # list of the supported FEM - For the moment only Q4 are supported
        #self.list_quantity_of_interest_to_plot = ['True strain','Deformation gradient','Engineering strain','Displacement','Coordinates','Green strain']
        self.list_camera_type = list(self.camera_type_dict.keys())
        # creation comboBox
        self.list_combo_camera_type=ttk.Combobox(setup_properties, values=self.list_camera_type,width=5)
        default_camera_type = 1
        self.list_combo_camera_type.current(default_camera_type)
        #Position de la ComboBox

        # Attribution of default value in case the user is satisfied with the proposed one
        self.list_combo_camera_type.bind("<<ComboboxSelected>>",self.select_quantity_of_interest_to_plot)

        self.list_combo_camera_type.grid(row=0, column=1, padx=2, pady=2)




        # Text area to define the prefix of the images
        smallest_size_text = 'Smallest sample size [mm]:'
        smallest_size = ttk.Label(setup_properties, text=smallest_size_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(smallest_size_text))
        smallest_size.grid(row=2, column=6, padx=2, pady=2)
        self.smallest_size_text_value = ttk.Entry(setup_properties,width=5)
        self.smallest_size_text_value.insert(0, 120)
        self.smallest_size_text_value.grid(row=2, column=7, padx=2, pady=2)


        # Text area to define the prefix of the images
        number_of_camera_text = 'Number of cameras:'
        number_of_camera = ttk.Label(setup_properties, text=number_of_camera_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(number_of_camera_text))
        number_of_camera.grid(row=0, column=8, padx=2, pady=2)
        self.number_of_camera_text_value = ttk.Entry(setup_properties,width=5)
        self.number_of_camera_text_value.insert(0, 1)
        self.number_of_camera_text_value.grid(row=0, column=9, padx=2, pady=2)

        # Text area to define the prefix of the images
        cover_pct_text = 'Overlap %age:'
        cover_pct = ttk.Label(setup_properties, text=cover_pct_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(cover_pct_text))
        cover_pct.grid(row=1, column=8, padx=2, pady=2)
        self.cover_pct_value = ttk.Entry(setup_properties,width=5)
        self.cover_pct_value.insert(0, 10)
        self.cover_pct_value.grid(row=1, column=9, padx=2, pady=2)

        self.cover_direction = {}
        self.cover_direction['Vert.']='Vertical'
        self.cover_direction['Hor.']='Horizontal'

        cover_direction_text = 'Overlap direction:'
        cover_direction = ttk.Label(setup_properties, text=cover_direction_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(cover_direction_text))
        cover_direction.grid(row=2, column=8, padx=2, pady=2)
       # list of the supported FEM - For the moment only Q4 are supported
        #self.list_quantity_of_interest_to_plot = ['True strain','Deformation gradient','Engineering strain','Displacement','Coordinates','Green strain']
        self.list_cover_direction = list(self.cover_direction.keys())
        # creation comboBox
        self.list_combo_cover_direction=ttk.Combobox(setup_properties, values=self.list_cover_direction,width=5)
        default_cover_direction = 1
        self.list_combo_cover_direction.current(default_cover_direction)
        #Position de la ComboBox

        # Attribution of default value in case the user is satisfied with the proposed one
        self.list_combo_cover_direction.bind("<<ComboboxSelected>>",self.select_cover_direction)

        self.list_combo_cover_direction.grid(row=2, column=9, padx=2, pady=2)





        distance_2_cams_text = '2 cameras dist. [m]:'
        distance_2_cams = ttk.Label(setup_properties, text=distance_2_cams_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(distance_2_cams_text))
        distance_2_cams.grid(row=2, column=10, padx=2, pady=2)

        self.distance_2_cams_value = ttk.Label(setup_properties, text='No value yet',foreground='blue')
        self.distance_2_cams_value.grid(row=2, column=11, padx=2, pady=2)        




        speckle_diam_text = 'Speckle size [mm]:'
        speckle_diam = ttk.Label(setup_properties, text=speckle_diam_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(speckle_diam_text))
        speckle_diam.grid(row=1, column=10, padx=2, pady=2)

        self.speckle_diam_value = ttk.Label(setup_properties, text='No value yet',foreground='blue')
        self.speckle_diam_value.grid(row=1, column=11, padx=2, pady=2)        






        nb_pixel_per_spot_text = 'Number of px per spot:'
        nb_pixel_per_spot = ttk.Label(setup_properties, text=nb_pixel_per_spot_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(nb_pixel_per_spot_text))
        nb_pixel_per_spot.grid(row=0, column=10, padx=2, pady=2)
        self.nb_pixel_per_spot_value = ttk.Entry(setup_properties,width=5)
        self.nb_pixel_per_spot_value.insert(0, 5)
        self.nb_pixel_per_spot_value.grid(row=0, column=11, padx=2, pady=2)



################################################################################################
        ######################## Frame content : test sketch ############################
################################################################################################

        #self.experimental_sketch.grid_columnconfigure(1, weight=1)
#        self.experimental_sketch.grid_rowconfigure(0, weight=0)

        #self.view_sketch = ttk.LabelFrame(self.experimental_sketch, text='Sketch of experimental setup')

        # field_to_plot.pack(anchor='n')
        # field_to_plot.place(in_=field_frame,
        #                     x=5,
        #                     # y=pos_vert_frame
        #                     )
        self.canvas_fig_setup_sketch = tk.Canvas(self.experimental_sketch)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.canvas_fig_setup_sketch.place(in_=self.experimental_sketch,height=24,width=18,relx=0.,x=2,rely=0.,y=2,anchor='n')

################################################################################################

        myFont = font.Font(weight="bold")
        calculate_test_recom_button = tk.Button(experimental_capabilities,text = "Compute test characteristics", command=self.compute_test_char,foreground='white',background='blue')
        calculate_test_recom_button['font'] = myFont
        calculate_test_recom_button.grid(row=0, column=0, padx=2, pady=2)




        # calculate_test_recom_button = tk.Button(experimental_capabilities, text="Compute test characteristics", command=self.compute_test_char)
        # calculate_test_recom_button.grid(row=0, column=0, padx=2, pady=2)






        observation_H_FOV_text = 'FOV size (H):'
        observation_H_FOV = ttk.Label(experimental_capabilities, text=observation_H_FOV_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(observation_H_FOV_text))
        observation_H_FOV.grid(row=0, column=1, padx=2, pady=2)
        self.H_FOV_value = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        self.H_FOV_value.grid(row=0, column=2, padx=2, pady=2)
        observation_V_FOV_text = '(V):'
        observation_V_FOV = ttk.Label(experimental_capabilities, text=observation_V_FOV_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(observation_V_FOV_text))
        observation_V_FOV.grid(row=0, column=3, padx=2, pady=2)

        self.V_FOV_value = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        self.V_FOV_value.grid(row=0, column=4, padx=2, pady=2)
############################################################################
        definition_FOV_text = 'Smallest sample def. [px]:'
        definition_FOV = ttk.Label(experimental_capabilities, text=definition_FOV_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(definition_FOV_text))
        definition_FOV.grid(row=1, column=2, padx=2, pady=2)
        self.definition_FOV_value = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        self.definition_FOV_value.grid(row=1, column=3, padx=2, pady=2)

###################################################################################################
        resolution_FOV_text = 'FOV res.:'
        resolution_FOV = ttk.Label(experimental_capabilities, text=resolution_FOV_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(resolution_FOV_text))
        resolution_FOV.grid(row=1, column=0, padx=2, pady=2)
        self.resolution_FOV_value = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        self.resolution_FOV_value.grid(row=1, column=1, padx=2, pady=2)
        # resolution_V_FOV_text = '(V):'
        # resolution_V_FOV = ttk.Label(experimental_capabilities, text=resolution_V_FOV_text,anchor='e',
        #                         #   width=len(text_object_label2)
        #                                 width=len(resolution_V_FOV_text))
        # resolution_V_FOV.grid(row=1, column=4, padx=2, pady=2)

        # self.resolution_V_FOV = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        # self.resolution_V_FOV.grid(row=1, column=5, padx=2, pady=2)        
###########################################################################################





        angle_of_view_H_text = 'Top angle of view [°]:'
        angle_of_view_H_lab = ttk.Label(experimental_capabilities, text=angle_of_view_H_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(angle_of_view_H_text))
        angle_of_view_H_lab.grid(row=0, column=5, padx=2, pady=2)

        self.angle_of_view_H_value = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        self.angle_of_view_H_value.grid(row=0, column=6, padx=2, pady=2)   

        angle_of_view_V_text = 'Side angle of view [°]:'
        angle_of_view_V_lab = ttk.Label(experimental_capabilities, text=angle_of_view_V_text+'°',anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(angle_of_view_V_text))
        angle_of_view_V_lab.grid(row=1, column=5, padx=2, pady=2)

        self.angle_of_view_V_value = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        self.angle_of_view_V_value.grid(row=1, column=6, padx=2, pady=2)   

        overlap_length_V_text = 'Overlap length (V) [m]:'
        overlap_length_V = ttk.Label(experimental_capabilities, text=overlap_length_V_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(overlap_length_V_text))
        overlap_length_V.grid(row=1, column=7, padx=2, pady=2)

        self.overlap_length_V_value = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        self.overlap_length_V_value.grid(row=1, column=8, padx=2, pady=2)   

        overlap_length_H_text = 'Overlap length (H) [m]:'
        overlap_length_H = ttk.Label(experimental_capabilities, text=overlap_length_H_text,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len(overlap_length_H_text))
        overlap_length_H.grid(row=0, column=7, padx=2, pady=2)

        self.overlap_length_H_value = ttk.Label(experimental_capabilities, text='No value yet',foreground='blue')
        self.overlap_length_H_value.grid(row=0, column=8, padx=2, pady=2)   

#####################################################################################
        ###################################################################
        # Code lines related to the first tab of the GUI / Pre-processing #
        ###################################################################
#####################################################################################

        # Frame pour user data
        preprocessing_frame = ttk.LabelFrame(tab1, text='Selection of the experimental raw images')
        # preprocessing_frame.pack(expand=1,fill='x',padx=2, pady=2)
        
        self.preview_first_image_frame = ttk.LabelFrame(tab1, text='Experimental field of view')
        # self.preview_first_image_frame.pack(expand=0, fill='x', padx=2, pady=2)

        ROI_frame = ttk.LabelFrame(tab1, text='Resizing the experimental images',height=100)
        # ROI_frame.pack(expand=1, fill='x', padx=2, pady=2)

        preprocessing_frame.grid(row=0, column=0, sticky="nswe")
        self.preview_first_image_frame.grid(row=1, column=0, sticky="nswe")
        ROI_frame.grid(row=2, column=0, sticky="nswe")


        #preprocessing_frame.grid_columnconfigure(0, weight=1)
        #preprocessing_frame.grid_rowconfigure(0,weight=1)


        chose_path_button = tk.Button(preprocessing_frame, text="1-Select folder", command=self.select_images_folder)
        chose_path_button.grid(row=0, column=0, padx=2, pady=2)

        chose_first_raw_file_button = tk.Button(preprocessing_frame, text="2-Select first image", command=self.select_first_image)
        chose_first_raw_file_button.grid(row=0, column=1, padx=2, pady=2)

        import_first_image_button = tk.Button(preprocessing_frame,text = "3-View first image", command = self.first_image_view)
        import_first_image_button.grid(row=0, column=2, padx=2, pady=2)


        # self.FRAME_for_icone=tk.Frame(frame_for_button_and_icone, width=10, height =10)
        # # self.FRAME_for_icone.place(in_=frame_for_button_and_icone, relx=1.0, x=-150)
        # self.FRAME_for_icone.pack(anchor='e',padx=80)

        # LABEL=tk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()







######################################################################################################
        # Frame showing the first images of the stack
######################################################################################################

        # Canvas pour afficher l'aperçu
        self.canvas_FOV_ROI = tk.Canvas(self.preview_first_image_frame)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.canvas_FOV_ROI.pack()


######################################################################################################
        # Frame for images cropping and generating of output cropped images
######################################################################################################
        


        select_ROI_button = tk.Button(ROI_frame,text = "Select zone to crop", command = self.select_ROI_rectangle)
        select_ROI_button.grid(row=0, column=4, padx=2, pady=2)

        # show_ROI_button = tk.Button(ROI_frame,text = "Apply ROI selection", command = self.plot_ROI_on_fig)
        # show_ROI_button.grid(row=0, column=1,columnspan=2, padx=2, pady=2)

        label_coord_corner1 = ttk.Label(ROI_frame, text='Coordinates in pixels of corner 1:')
        label_coord_corner1.grid(row=0, column=0, padx=2, pady=2)
        self.value_coord_corner1 = ttk.Label(ROI_frame, text='No coord. yet',foreground='blue')
        self.value_coord_corner1.grid(row=0, column=1, padx=2, pady=2)
        
        # self.prefix_entry.insert(0, 'Gauche_Droite-')
        # self.prefix_entry.grid(row=1, column=3, padx=2, pady=2)

        label_coord_corner2 = ttk.Label(ROI_frame, text='Coordinates in pixels of corner 2:')
        label_coord_corner2.grid(row=0, column=2, padx=4, pady=2)
        self.value_coord_corner2 = ttk.Label(ROI_frame, text='No coord. yet',foreground='blue')
        self.value_coord_corner2.grid(row=0, column=3, padx=2, pady=2)

        # Text area to define the prefix of the images
        prefix_ROI_files = ttk.Label(ROI_frame, text='User name for images:')
        prefix_ROI_files.grid(row=1, column=0, padx=2, pady=2)
        self.prefix_ROI_files_entry = ttk.Entry(ROI_frame)
        self.prefix_ROI_files_entry.insert(4, 'cropped_view_dynamic_beam_test_')
        self.prefix_ROI_files_entry.grid(row=1, column=1, padx=2, pady=2)

        output_ROI_dir = ttk.Label(ROI_frame, text='Name of the local output directory:')
        output_ROI_dir.grid(row=1, column=0, padx=2, pady=2)
        self.output_ROI_dir_entry = ttk.Entry(ROI_frame)
        self.output_ROI_dir_entry.insert(4, 'Cropped_Images')
        self.output_ROI_dir_entry.grid(row=1, column=1, padx=2, pady=2)


        # prepare_resize_ROI_button2 = tk.Button(ROI_frame,text = "Number of files to be cropped", command=get_number_of_files())
        # prepare_resize_ROI_button2.grid(row=2, column=3,columnspan=2, padx=2, pady=2)

        # Text area to define the prefix of the images
        prepare_resize_ROI_files = ttk.Label(ROI_frame, text='Number of files to crop:')
        prepare_resize_ROI_files.grid(row=1, column=2, padx=2, pady=2)
        self.prepare_resize_ROI_files_entry = ttk.Entry(ROI_frame,width=6)
        self.prepare_resize_ROI_files_entry.insert(0, 2)
        self.prepare_resize_ROI_files_entry.grid(row=1, column=3, padx=2, pady=2)


        text_num_images = 'Numbering format of the images \n (ex: if 00 selected then 00u, 0du ...):'
        # Menu to select the numbering of the stack of images
        num_images = ttk.Label(ROI_frame, text=text_num_images,anchor='e',justify='right',width=len('(ex: if 00 selected then 00u, 0du ...):'))
        num_images.grid(row=2, column=0, padx=2, pady=2)

        # list of the supported numbering types
        self.list_num_images = ['0','00','000','0000','00000','000000']
        # creation comboBox
        self.list_combo_num_images=ttk.Combobox(ROI_frame, values=self.list_num_images,width=6)
        default_num_image = 3
        self.list_combo_num_images.current(default_num_image)
        #ComboBox location
        self.list_combo_num_images.grid(row=2, column=1, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.num_images = self.list_num_images[default_num_image]
        self.list_combo_num_images.bind("<<ComboboxSelected>>",self.select_menu_num_images)


        # Menu to select the numbering of the stack of images
        FEM_text_format_image = 'Type of the images:'
        FEM_format_image = ttk.Label(ROI_frame, text=FEM_text_format_image,anchor='e',width=len('per Finite Element side:')
                                #  ,width=len(text_format_image)
                                 )
        FEM_format_image.grid(row=2, column=2, padx=2, pady=2)



        # list of the supported numbering types
        self.list_format_image = ['.tif','.tiff','.png']
        # creation comboBox
        self.list_combo_format_image=ttk.Combobox(ROI_frame, values=self.list_format_image,width=5)
        default_format_image = 1
        self.list_combo_format_image.current(default_format_image)
        #Position de la ComboBox
        self.list_combo_format_image.grid(row=2, column=3, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.format_image = self.list_format_image[default_format_image]
        self.list_combo_format_image.bind("<<ComboboxSelected>>",self.select_menu_format_image)





        #prepare_resize_ROI_button = tk.Button(ROI_frame,text = "Generate default output directory", command=lambda:[self.create_output_folder, get_number_of_files()])
        prepare_resize_ROI_button1 = tk.Button(ROI_frame,text = "Create output dir.", command=self.create_output_folder)
        prepare_resize_ROI_button1.grid(row=1, column=4, padx=2, pady=2)

        myFont = font.Font(weight="bold")
        resize_ROI_button = tk.Button(ROI_frame,text = "Export", command=self.crop_images_to_ROI,foreground='white',background='blue')
        resize_ROI_button['font'] = myFont
        resize_ROI_button.grid(row=2, column=4, padx=2, pady=2)


##################################################################################################
        ###############################################################################
        # Code lines related to the second tab of the GUI / Digital Image Correlation #
        ###############################################################################
##################################################################################################



        FEM_frame = ttk.LabelFrame(tab2, text='Experimental Images View and Stacking')
        FEM_frame.grid(row=0, column=0, sticky="nswe")
        # FEM_frame.pack(expand=1, fill='x', padx=2, pady=2)

        # self.FRAME_for_icone=ttk.Frame(FEM_frame, width=10, height =10)
        # self.FRAME_for_icone.place(in_=FEM_frame, relx=1.0, x=-150, rely=0, y=5)

        # LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()




        FEM_chose_path_button = tk.Button(FEM_frame, text="1-Select folder", command=self.FEM_select_images_folder)
        FEM_chose_path_button.grid(row=0, column=0, padx=2, pady=2)

        FEM_chose_first_raw_file_button = tk.Button(FEM_frame, text="2-Select first image", command=self.FEM_select_first_image)
        FEM_chose_first_raw_file_button.grid(row=0, column=1, padx=2, pady=2)

        FEM_import_first_image_button = tk.Button(FEM_frame,text = "3-View first image", command = self.FEM_first_image_view)
        FEM_import_first_image_button.grid(row=0, column=2, padx=2, pady=2)



        
        # Text area to define the prefix of the images
        FEM_nb_stack_images = '# of images to stack:'
        FEM_nb_stack_images = ttk.Label(FEM_frame, text=FEM_nb_stack_images,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len('per Finite Element side:'))
        FEM_nb_stack_images.grid(row=0, column=3, padx=2, pady=2)
        self.FEM_nb_stack_images = ttk.Entry(FEM_frame,width=5)
        self.FEM_nb_stack_images.insert(0, 10)

        self.FEM_nb_stack_images.grid(row=0, column=4, padx=2, pady=2)

        FEM_text_chose_path_button = "4-Stack images"
        FEM_chose_path_button = tk.Button(FEM_frame, text=FEM_text_chose_path_button, command=self.image_stacking
                                    #    , width=len(text_chose_path_button)
                                       )
        FEM_chose_path_button.grid(row=0, column=5, padx=2, pady=2)
        
        FEM_select_ROI_button = tk.Button(FEM_frame,text = "5-Define ROI on screen", command = self.FEM_select_ROI_rectangle)
        FEM_select_ROI_button.grid(row=0, column=6, padx=2, pady=2)











#########################################################################################
    # Frame for vizualizing the ROI on the cropped images
#########################################################################################
        self.FEM_preview_first_image_frame = ttk.LabelFrame(tab2, text='ROI selection and meshing')
        # self.FEM_preview_first_image_frame.pack(expand=0, fill='x', padx=2, pady=2)
        self.FEM_preview_first_image_frame.grid(row=1, column=0, sticky="nswe")
        # Canvas pour afficher l'aperçu
        self.FEM_canvas_FOV_ROI = tk.Canvas(self.FEM_preview_first_image_frame)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.FEM_canvas_FOV_ROI.pack()

#############################################################################################
    # Frame for ROI construction and mesh generation
##############################################################################################

        FEM_ROI_frame = ttk.LabelFrame(tab2, text='Definition of the ROI')
        FEM_ROI_frame.grid(row=2, column=0, sticky="nswe")
        # pack(expand=1, fill='x', padx=2, pady=2)


        # show_ROI_button = tk.Button(ROI_frame,text = "Apply ROI selection", command = self.plot_ROI_on_fig)
        # show_ROI_button.grid(row=0, column=1,columnspan=2, padx=2, pady=2)

        FEM_label_coord_corner1 = ttk.Label(FEM_ROI_frame, text='Coordinates in pixels of corner 1:')
        FEM_label_coord_corner1.grid(row=0, column=0, padx=2, pady=2)
        self.FEM_value_coord_corner1 = ttk.Label(FEM_ROI_frame, text='No coord. yet',foreground='blue')
        self.FEM_value_coord_corner1.grid(row=0, column=1, padx=2, pady=2)
        
        # self.prefix_entry.insert(0, 'Gauche_Droite-')
        # self.prefix_entry.grid(row=1, column=3, padx=2, pady=2)

        FEM_label_coord_corner2 = ttk.Label(FEM_ROI_frame, text='Coordinates in pixels of corner 2:')
        FEM_label_coord_corner2.grid(row=0, column=2, padx=4, pady=2)
        self.FEM_value_coord_corner2 = ttk.Label(FEM_ROI_frame, text='No coord. yet',foreground='blue')
        self.FEM_value_coord_corner2.grid(row=0, column=3, padx=2, pady=2)


        # Menu to select the type of Finite Element
        text_FEM_type = 'Finite Element type:'
        FEM_type = ttk.Label(FEM_ROI_frame, text=text_FEM_type,anchor='e',width=len('Finite Element type:')
                            #  , width=len(text_FEM_type)
                             )
        FEM_type.grid(row=1, column=0, padx=2, pady=2)
       # list of the supported FEM - For the moment only Q4 are supported
        self.list_FEM_type = ['Q4','T3']
        # creation comboBox
        self.list_combo_FEM_type=ttk.Combobox(FEM_ROI_frame, values=self.list_FEM_type,width=5)
        default_FEM_type = 0
        self.list_combo_FEM_type.current(default_FEM_type)
        #Position de la ComboBox
        self.list_combo_FEM_type.grid(row=1, column=1, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.FEM_type = self.list_FEM_type[default_FEM_type]
        self.list_combo_FEM_type.bind("<<ComboboxSelected>>",self.select_menu_FEM_type)




        # Text area to define the number of pixels per Finite Element
        text_nb_pix_per_elem_label = 'Number of pixels\n per Finite Element side:'
        nb_pix_per_elem_label = ttk.Label(FEM_ROI_frame, text=text_nb_pix_per_elem_label,anchor='e',
                                           width=len('per Finite Element side:')
                                          )
        nb_pix_per_elem_label.grid(row=1, column=2, padx=2, pady=2)
        self.nb_pix_per_elem_entry = ttk.Entry(FEM_ROI_frame,width=6)
        self.nb_pix_per_elem_entry.insert(0, 20)
        self.nb_pix_per_elem_entry.grid(row=1, column=3, padx=2, pady=2)


        self.nb_pix_per_elem = np.int64(self.nb_pix_per_elem_entry.get())
        # nb_elx = np.int64(np.floor((xmax-xmin)/nb_pix_per_elem))
        # nb_ely = np.int64(np.floor((ymax-ymin)/nb_pix_per_elem))



        FEM_text_mesh_prop_button = "Generate FEM mesh properties"
        FEM_mesh_prop_button = tk.Button(FEM_ROI_frame, text=FEM_text_mesh_prop_button, command=self.FEM_mesh_prop_gen
                                    #    , width=len(text_chose_path_button)
                                       )
        FEM_mesh_prop_button.grid(row=0, column=4,  padx=2, pady=2)

        FEM_generate_button = tk.Button(FEM_ROI_frame,text = "Create mesh", command = self.FEM_generate)
        FEM_generate_button.grid(row=1, column=4, padx=2, pady=2)


#########################################################################################

        DIC_solver_frame = ttk.LabelFrame(tab2, text='DIC solver')
        DIC_solver_frame.grid(row=3, column=0, sticky="nswe")
        # DIC_solver_frame.pack(expand=1, fill='x', padx=2, pady=2)

        first_image_for_DIC = ttk.Label(DIC_solver_frame, text='First (reference) image of the DIC solver:')
        first_image_for_DIC.grid(row=0, column=0, padx=2, pady=2)
        self.first_image_for_DIC_entry = ttk.Entry(DIC_solver_frame,width=6)
        self.first_image_for_DIC_entry.insert(0, 1)
        self.first_image_for_DIC_entry.grid(row=0, column=1, padx=2, pady=2)

        last_image_for_DIC = ttk.Label(DIC_solver_frame, text='Last stacked image:')
        last_image_for_DIC.grid(row=0, column=2, padx=2, pady=2)
        self.last_image_for_DIC_entry = ttk.Entry(DIC_solver_frame,width=6)
        self.last_image_for_DIC_entry.insert(0, 4)
        self.last_image_for_DIC_entry.grid(row=0, column=3, padx=2, pady=2)



        FEM_generate_settings = tk.Button(DIC_solver_frame,text = "Generate DIC settings", command = self.DIC_generate_settings)
        FEM_generate_settings.grid(row=0, column=6, padx=2, pady=2)


        iteration = ttk.Label(DIC_solver_frame, text='Maximum iterations:')
        iteration.grid(row=1, column=0, padx=2, pady=2)
        self.iteration_entry = ttk.Entry(DIC_solver_frame,width=6)
        self.iteration_entry.insert(0, 20)
        self.iteration_entry.grid(row=1, column=1, padx=2, pady=2)

        convergence = ttk.Label(DIC_solver_frame, text='Convergence tolerance:')
        convergence.grid(row=1, column=2, padx=2, pady=2)
        self.convergence = ttk.Entry(DIC_solver_frame,width=6)
        self.convergence.insert(0, 1E-6)
        self.convergence.grid(row=1, column=3, padx=2, pady=2)

        order_interp = ttk.Label(DIC_solver_frame, text='Interpolation order:')
        order_interp.grid(row=1, column=4, padx=2, pady=2)
        self.order_interp = ttk.Entry(DIC_solver_frame,width=6)
        self.order_interp.insert(0, 2)
        self.order_interp.grid(row=1, column=5, padx=2, pady=2)


        # Menu to select the type of Finite Element
        store_internal_var = 'Store internal variables:'
        store_internal_var = ttk.Label(DIC_solver_frame, text=store_internal_var,anchor='e',width=len('Store internal variables:')
                            #  , width=len(text_FEM_type)
                             )
        store_internal_var.grid(row=0, column=4, padx=2, pady=2)
       # list of the supported FEM - For the moment only Q4 are supported
        self.list_store_internal_var = ['Yes','No']
        # creation comboBox
        self.list_combo_store_internal_var=ttk.Combobox(DIC_solver_frame, values=self.list_store_internal_var,width=5)
        default_store_internal_var = 0
        self.list_combo_store_internal_var.current(default_store_internal_var)
        #Position de la ComboBox
        self.list_combo_store_internal_var.grid(row=0, column=5, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.temp_store_internals = True
        self.list_combo_store_internal_var.bind("<<ComboboxSelected>>",self.select_menu_store_internal_var)

        prepare_DIC_analysis = tk.Button(DIC_solver_frame,text = "Prepare analysis", command = self.prepare_DIC_analysis)
        prepare_DIC_analysis.grid(row=1, column=6, padx=2, pady=2)





        myFont = font.Font(weight="bold")
        launch_DIC_analysis = tk.Button(DIC_solver_frame,text = "Run "+self.text_muDIC, bg='Blue', fg='White', command = self.run_DIC_analysis)
        # "Run "+u"\N{GREEK SMALL LETTER MU}"+"DIC"
        # Define font
        launch_DIC_analysis['font'] = myFont
        launch_DIC_analysis.grid(row=2, column=6, padx=2, pady=2)


        no_convergence_options = 'Action if no convergence:'
        no_convergence_options = ttk.Label(DIC_solver_frame, text=no_convergence_options,anchor='e',width=len('Action if no convergence:')
                            #  , width=len(text_FEM_type)
                             )
        no_convergence_options.grid(row=2, column=0, padx=2, pady=2)
       # list of the supported FEM - For the moment only Q4 are supported
        self.list_no_convergence_options = ["ignore", "update", "break"]
        # creation comboBox
        self.list_combo_no_convergence_options=ttk.Combobox(DIC_solver_frame, values=self.list_no_convergence_options,width=5)
        default_no_convergence_options = 0
        self.list_combo_no_convergence_options.current(default_no_convergence_options)
        #Position de la ComboBox
        self.list_combo_no_convergence_options.grid(row=2, column=1, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.list_combo_no_convergence_options.bind("<<ComboboxSelected>>",self.select_menu_no_convergence_options)


#####################################################################################
        ####################################################################
        # Code lines related to the third tab of the GUI / Post-processing #
        ####################################################################
#####################################################################################


        field_frame = ttk.LabelFrame(tab3, text='Field of quantities of interest')
        # field_frame.pack(expand=1, fill='x', padx=2, pady=2)
        field_frame.grid(row=0, column=0, sticky='nswe')
       
        field_frame.grid_columnconfigure(1, weight=1)
        field_frame.grid_rowconfigure(0, weight=1)
        # field_frame.grid_columnconfigure(0, weight=1)
        # field_frame.grid_columnconfigure(1, weight=1)
        # field_frame.grid_rowconfigure(0, weight=1)
        # field_frame.grid_rowconfigure(1, weight=1)

        # self.FRAME_for_icone=ttk.Frame(field_frame, width=10, height =10)
        # self.FRAME_for_icone.place(in_=field_frame, relx=1.0, x=-150, rely=0, y=5)

        # LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()

        # pos_vert_frame = 1
        field_to_plot = ttk.LabelFrame(field_frame, text='Quantity of interest - Units')
        # field_to_plot.pack(anchor='n')
        # field_to_plot.place(in_=field_frame,
        #                     x=5,
        #                     # y=pos_vert_frame
        #                     )

        frame_to_plot = ttk.LabelFrame(field_frame, text='Frame selection')
        # frame_to_plot.pack(anchor='n')
        # frame_to_plot.place(in_=field_frame,
        #                     x=5,
        #                     y=140
        #                     # y=140-pos_vert_frame
        #                     )


        range_to_plot = ttk.LabelFrame(field_frame, text='Range')
        # range_to_plot.pack(anchor='ne')
        # range_to_plot.place(in_=field_frame,
        #                     # y=pos_vert_frame,
        #                     x=610
        #                     )
        field_to_plot.grid(row=0, column=0, sticky="ns")
        frame_to_plot.grid(row=0, column=1, sticky="nsew")
        range_to_plot.grid(row=0, column=2, sticky="ns")



        frame_for_plot_button=ttk.Frame(range_to_plot, width=10, height =10)
        frame_for_plot_button.place(in_=range_to_plot,rely=1,y=-30,relx=1,x=-50)


############################################################################################
        self.fields = {}
        self.fields['True strain']='true_strain()'
        self.fields['Deformation gradient']='F()'
        self.fields['Engineering strain']='eng_strain()'
        self.fields['Displacement']='disp()'
        self.fields['Coordinates']='coords()'
        self.fields['Green strain']='green_strain()'
        
        text_quantity_of_interest_to_plot = 'Quantity of interest:'
        quantity_of_interest_to_plot = ttk.Label(field_to_plot, text=text_quantity_of_interest_to_plot,anchor='e',width=len(text_quantity_of_interest_to_plot)
                             )
       # list of the supported FEM - For the moment only Q4 are supported
        #self.list_quantity_of_interest_to_plot = ['True strain','Deformation gradient','Engineering strain','Displacement','Coordinates','Green strain']
        self.list_quantity_of_interest_to_plot = list(self.fields.keys())
        # creation comboBox
        self.list_combo_quantity_of_interest_to_plot=ttk.Combobox(field_to_plot, values=self.list_quantity_of_interest_to_plot,width=len('Deformation gradient'))
        default_quantity_of_interest_to_plot = 3
        self.list_combo_quantity_of_interest_to_plot.current(default_quantity_of_interest_to_plot)
        #Position de la ComboBox

        quantity_of_interest_to_plot.grid(row=0,column=0, padx=2, pady=2)
        self.list_combo_quantity_of_interest_to_plot.grid(row=0, column=1, padx=2, pady=2)


        # Attribution of default value in case the user is satisfied with the proposed one
        self.list_combo_quantity_of_interest_to_plot.bind("<<ComboboxSelected>>",self.select_quantity_of_interest_to_plot)

        self.component = {}
        self.component['Disp-x']=(0,0)
        self.component['Disp-y']=(1,0)
        self.component['xx']=(0,0)
        self.component['xy']=(0,1)
        self.component['yy']=(1,1)
       
        text_component_quantity_of_interest_to_plot = 'Component to plot:'
        component_quantity_of_interest_to_plot = ttk.Label(field_to_plot, text=text_component_quantity_of_interest_to_plot,anchor='e',width=len(text_component_quantity_of_interest_to_plot)
                             )
       # list of the supported FEM - For the moment only Q4 are supported
        self.list_component_quantity_of_interest_to_plot = list(self.component.keys())
        #['Disp-x','Disp-y','xx','xy','yy']
        # creation comboBox
        self.list_combo_component_quantity_of_interest_to_plot=ttk.Combobox(field_to_plot, values=self.list_component_quantity_of_interest_to_plot,width=len('Disp-x'))
        default_component_quantity_of_interest_to_plot = 0
        self.list_combo_component_quantity_of_interest_to_plot.current(default_component_quantity_of_interest_to_plot)
        #Position de la ComboBox



        component_quantity_of_interest_to_plot.grid(row=1, column=0, padx=2, pady=2)
        self.list_combo_component_quantity_of_interest_to_plot.grid(row=1, column=1, padx=2, pady=2)


        # Attribution of default value in case the user is satisfied with the proposed one
        self.list_combo_component_quantity_of_interest_to_plot.bind("<<ComboboxSelected>>",self.select_component_quantity_of_interest_to_plot)




        self.type_of_unit= tk.IntVar(value=1)

        ttk.Radiobutton(field_to_plot,text='Pixels as length unit',variable=self.type_of_unit, value=1).grid(row=2, column=0, padx=2, pady=2)
        ttk.Radiobutton(field_to_plot,text='Actual length unit',variable=self.type_of_unit, value=2).grid(row=2, column=1, padx=2, pady=2)

        self.units = {}
        self.units['m']='m'
        self.units['cm']='cm'
        self.units['mm']='mm'

        text_units = 'Unit:'
        text_units = ttk.Label(field_to_plot, text=text_units,anchor='e',width=len(text_units)
                             )
        text_units.grid(row=3, column=0, padx=2, pady=2)
       # list of the supported FEM - For the moment only Q4 are supported
        self.list_text_units = list(self.units.keys())
        #['Disp-x','Disp-y','xx','xy','yy']
        # creation comboBox
        self.list_combo_text_units=ttk.Combobox(field_to_plot, values=self.list_text_units,width=len('mm    '))
        default_text_units = 0
        self.list_combo_text_units.current(default_text_units)
        #Position de la ComboBox
        self.list_combo_text_units.grid(row=3, column=1, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.list_combo_text_units.bind("<<ComboboxSelected>>",self.select_text_units)



        pick_ref_length_object = tk.Button(field_to_plot,text = "Pick ref. on-screen", command = self.select_ref_length)
        pick_ref_length_object.grid(row=5, column=0, padx=2, pady=2)


        label_ref_length = ttk.Label(field_to_plot, text='Ref. length:')
        label_ref_length.grid(row=4, column=0, padx=2, pady=2)

        self.value_ref_length = ttk.Entry(field_to_plot,width=5)
        self.value_ref_length.insert(0, 0.15)
        self.value_ref_length.grid(row=4, column=1, padx=2, pady=2)

        label_scale_DIC = ttk.Label(field_to_plot, text='Scale value:')

        label_scale_DIC.grid(row=6, column=0, padx=2, pady=2)
        self.value_scale_DIC = ttk.Label(field_to_plot, text='No scale yet',foreground='blue')

        self.value_scale_DIC.grid(row=6, column=1, padx=2, pady=2)







################################################
        # ttk.Radiobutton(frame_to_plot,text='All frames',variable=self.frames_to_be_plotted, value=1).grid(row=0, column=6, padx=2, pady=2)
        # ttk.Radiobutton(frame_to_plot,text='Range of frames',variable=self.frames_to_be_plotted, value=2).grid(row=1, column=6, padx=2, pady=2)



        self.frames_to_be_plotted = tk.IntVar(value=1)


        ttk.Radiobutton(frame_to_plot,text='All frames',variable=self.frames_to_be_plotted, value=1).grid(row=0, column=0, padx=2, pady=2)
        ttk.Radiobutton(frame_to_plot,text='Range of frames',variable=self.frames_to_be_plotted, value=2).grid(row=1, column=0, padx=2, pady=2)
        ttk.Radiobutton(frame_to_plot,text='Individual frame',variable=self.frames_to_be_plotted, value=3).grid(row=4, column=0, padx=2, pady=2)


        # button_one_frame = ttk.Radiobutton(field_frame,'Individual frame')
        # button_set_of_frames = ttk.Radiobutton(field_frame,'Range of frames')
        # button_view_all_frames = ttk.Radiobutton(field_frame,'All frames')

        first_frame_to_plot = 'First frame # to plot:'
        first_frame_to_plot = ttk.Label(frame_to_plot, text=first_frame_to_plot,anchor='e',width=len('Frame # to plot::')
                            #  , width=len(text_FEM_type)
                             )
        first_frame_to_plot.grid(row=2, column=0, padx=2, pady=2)
        if self.DIC_settings is not None:
            spinbox_first_frame_to_view = ttk.Spinbox(frame_to_plot, from_=1, to=int(np.int64(self.FEM_nb_stack_images.get())),width=6)
        else:
            spinbox_first_frame_to_view = ttk.Spinbox(frame_to_plot, from_=1, to=2,width=6)
        spinbox_first_frame_to_view.grid(row=2, column=1)



        last_frame_to_plot = 'Last frame # to plot:'
        last_frame_to_plot = ttk.Label(frame_to_plot, text=last_frame_to_plot,anchor='e',width=len('Frame # to plot::')
                            #  , width=len(text_FEM_type)
                             )
        last_frame_to_plot.grid(row=3, column=0, padx=2, pady=2)
        if self.DIC_settings is not None:
            spinbox_last_frame_to_view = ttk.Spinbox(frame_to_plot, from_=1, to=int(np.int64(self.FEM_nb_stack_images.get())),width=6)
        else:
            spinbox_last_frame_to_view = ttk.Spinbox(frame_to_plot, from_=1, to=2,width=6)
        spinbox_last_frame_to_view.grid(row=3, column=1)



        frame_number_to_plot = 'Frame # to plot:'
        frame_number_to_plot = ttk.Label(frame_to_plot, text=frame_number_to_plot,anchor='e',width=len('Frame # to plot::')
                            #  , width=len(text_FEM_type)
                             )
        frame_number_to_plot.grid(row=5, column=0, padx=2, pady=2)
        
        if self.DIC_settings is not None:
            spinbox_frame_to_view = ttk.Spinbox(frame_to_plot, from_=1, to=int(np.int64(self.FEM_nb_stack_images.get())),width=6)
        else:
            spinbox_frame_to_view = ttk.Spinbox(frame_to_plot, from_=1, to=2,width=6)
        spinbox_frame_to_view.grid(row=5, column=1)

        # Creating a dict with the correspondance between the keywords of the fields that 
        # are implemented in terms immediatly meanful for the user
        row_pos_custom = 3
        col_pos_custom = 0
        self.range_view = tk.IntVar(value=1)
        ttk.Radiobutton(range_to_plot,text='Static range of min-max values',variable=self.range_view, value=1).grid(row=row_pos_custom-2, column=col_pos_custom, padx=2, pady=2)
        ttk.Radiobutton(range_to_plot,text='Custom range of min-max values',variable=self.range_view, value=3).grid(row=row_pos_custom, column=col_pos_custom, padx=2, pady=2)
        ttk.Radiobutton(range_to_plot,text='Dynamic range of min-max values',variable=self.range_view, value=2).grid(row=row_pos_custom-1, column=col_pos_custom, padx=2, pady=2)
        
        range_min_value = ttk.Label(range_to_plot, text='Custom min:',anchor='center',width=len('Custom min:   ')
                            #  , width=len(text_FEM_type)
                             )
        col_pos_min = col_pos_custom+1
        col_pos_max = col_pos_custom+2
        range_min_value.grid(row=row_pos_custom+1, column=col_pos_min-1, padx=4, pady=2)
        # range_min_value.place(in_=range_to_plot,x=5,y=100)
        range_max_value = ttk.Label(range_to_plot, text='Custom max:',anchor='center',width=len('Custom max:   ')
                            #  , width=len(text_FEM_type)
                             )

        range_max_value.grid(row=row_pos_custom+1, column=col_pos_max-1, padx=4, pady=2)
        #range_max_value.place(in_=range_to_plot,x=200,y=100)
        
        
        
        if self.DIC_results is not None:
            scale_min_value = tk.Scale(range_to_plot,from_=self.min_value_quantity_of_interest,to=self.max_value_quantity_of_interest,
                                        tickinterval=1000,
                                        orient='horizontal',
                                        sliderlength=len('Custom max:   '),
                                        )
            # ttk.Spinbox(frame_to_plot, from_=0, to=self.min_value_quantity_of_interest,width=6)
            scale_max_value = tk.Scale(range_to_plot, from_=self.min_value_quantity_of_interest, to=self.max_value_quantity_of_interest,
                                        tickinterval=1000,
                                        orient='horizontal',
                                        sliderlength=len('Custom max:   '),
                                        )

        else:
            scale_min_value = tk.Scale(range_to_plot,from_=-10.,to=0.,
                                        tickinterval=1000,
                                        orient='horizontal',
                                        sliderlength=len('Custom min:   '),
                                        )

            # ttk.Spinbox(frame_to_plot, from_=0, to=self.min_value_quantity_of_interest,width=6)
            scale_max_value = tk.Scale(range_to_plot, from_=-10., to=0.,
                                        tickinterval=1000,
                                        orient='horizontal',
                                        sliderlength=len('Custom max:   '),
                                        )
        # scale_min_value.place(in_=range_to_plot,x=5+len('Min   ')+5,y=100)
        scale_min_value.grid(row=row_pos_custom+2, column=col_pos_min-1, padx=4, pady=2)
        scale_max_value.grid(row=row_pos_custom+2, column=col_pos_max-1, padx=4, pady=2)


        myFont = font.Font(weight="bold")
        visualize_field = tk.Button(range_to_plot,text = "Plot fringe", bg='Blue', fg='White', command = self.plot_fringe)
        # "Run "+u"\N{GREEK SMALL LETTER MU}"+"DIC"
        # Define font
        visualize_field['font'] = myFont
        visualize_field.grid(row=row_pos_custom+3,column=col_pos_max-1,padx=2,pady=2)

        self.view_postpro_local_and_fields = ttk.LabelFrame(tab3, text='Post-processing viewer')
        self.view_postpro_local_and_fields.grid(row=1, column=0, sticky='nswe')
        # self.view_postpro_local_and_fields.pack(expand=0, fill='x', padx=2, pady=2)





##############################################
        # Canvas pour afficher l'aperçu
        self.canvas_view_postpro = tk.Canvas(self.view_postpro_local_and_fields)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.canvas_view_postpro.pack()




##########################################################################################################
####################### Tab for local post-processing - plotting disp, vel and acc along time ############
####################### For dynamic test => add SDOF calc, FFT and CCWT and EMA and OMA ##################
##########################################################################################################

        # local_measurement_frame.pack(expand=1, fill='x', padx=2, pady=2)
        self.view_postpro_local = ttk.LabelFrame(tab3bis, text='Post-processing viewer')
        self.view_postpro_local.grid(row=0, column=0, sticky='nswe')
        # self.view_postpro_local_and_fields.pack(expand=0, fill='x', padx=2, pady=2)

        # Canvas pour afficher l'aperçu
        self.canvas_view_postpro_local = tk.Canvas(self.view_postpro_local)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.canvas_view_postpro_local.pack()



        local_measurement_frame = ttk.LabelFrame(tab3bis, text='Local measurements')
        local_measurement_frame.grid(row=1, column=0, sticky='nswe')


        local_measurement_frame.grid_columnconfigure(1, weight=1)
        local_measurement_frame.grid_rowconfigure(0, weight=1)
        # field_frame.grid_columnconfigure(0, weight=1)
        # field_frame.grid_columnconfigure(1, weight=1)
        # field_frame.grid_rowconfigure(0, weight=1)
        # field_frame.grid_rowconfigure(1, weight=1)

        # self.FRAME_for_icone=ttk.Frame(field_frame, width=10, height =10)
        # self.FRAME_for_icone.place(in_=field_frame, relx=1.0, x=-150, rely=0, y=5)

        # LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()

        # pos_vert_frame = 1

        quasistatic_analysis = ttk.LabelFrame(local_measurement_frame, text='Quasi-static analysis')
        # field_to_plot.pack(anchor='n')
        # field_to_plot.place(in_=field_frame,
        #                     x=5,
        #                     # y=pos_vert_frame
        #                     )
        pick_a_point_button_text = "Pick location"
        pick_a_point_button = tk.Button(quasistatic_analysis, text=pick_a_point_button_text, command=self.FEM_mesh_prop_gen
                                    #    , width=len(text_chose_path_button)
                                       )
        pick_a_point_button.grid(row=0, column=0,  padx=2, pady=2)


        plot_curve_text = "Plot curve"
        plot_curve_button = tk.Button(quasistatic_analysis, text=plot_curve_text, command=self.plot_fringe
                                    #    , width=len(text_chose_path_button)
                                       )
        plot_curve_button.grid(row=1, column=0,  padx=2, pady=2)



        time_history_analysis = ttk.LabelFrame(local_measurement_frame, text='Time-history analysis')
        # frame_to_plot.pack(anchor='n')
        # frame_to_plot.place(in_=field_frame,
        #                     x=5,
        #                     y=140
        #                     # y=140-pos_vert_frame
        #                     )

        plot_time_hist_text = "Plot time-history"
        plot_time_hist_button = tk.Button(time_history_analysis, text=plot_time_hist_text, command=self.plot_fringe
                                    #    , width=len(text_chose_path_button)
                                       )
        plot_time_hist_button.grid(row=0, column=0,  padx=2, pady=2)




        frequency_domain_analysis = ttk.LabelFrame(local_measurement_frame, text='Frequency domain analysis')
        # range_to_plot.pack(anchor='ne')
        # range_to_plot.place(in_=field_frame,
        #                     # y=pos_vert_frame,
        #                     x=610
        #                     )
        plot_SDOF_text = "Plot SDOF"
        plot_SDOF_button = tk.Button(frequency_domain_analysis, text=plot_SDOF_text, command=self.plot_fringe
                                    #    , width=len(text_chose_path_button)
                                       )
        plot_SDOF_button.grid(row=0, column=0,  padx=2, pady=2)


        time_frequency_domain_analysis = ttk.LabelFrame(local_measurement_frame, text='Time-frequency domain analysis')
        # range_to_plot.pack(anchor='ne')
        # range_to_plot.place(in_=field_frame,
        #                     # y=pos_vert_frame,
        #                     x=610
        #                     )

        plot_CCWT_text = "Plot CCWT"
        plot_CCWT_button = tk.Button(time_frequency_domain_analysis, text=plot_CCWT_text, command=self.plot_fringe
                                    #    , width=len(text_chose_path_button)
                                       )
        plot_CCWT_button.grid(row=0, column=0,  padx=2, pady=2)


        quasistatic_analysis.grid(row=0, column=0, sticky="nswe")
        time_history_analysis.grid(row=0, column=1, sticky="nswe")
        frequency_domain_analysis.grid(row=0, column=2, sticky="nswe")
        time_frequency_domain_analysis.grid(row=0, column=3, sticky="nswe")













#############################################################################################


        """
        Ajouter une mesure de l'incertitude de mesure avec écart type sur une zone non sollicitée et aussi hors excitation
        Ajouter une sélection à l'écran pour calculer le paramètre d'échelle pixel - dimension physique
        Ajouter dans tous les graphiques la toolbar de matplotlib
        Ajouter un bouton pour lancer l'extraction des quantités d'intérêt globale
        Ajouter un sélecteur pour choisir si on veut extraire pour tous les instants ou pour un seul instant
        Ajouter un bouton pour calculer les quantités d'intérêt choisies
        Ajouter un bouton pour lancer la visualisation avec les options ci-dessous
        Pour la visualisation sur tous les instants choisir dynamic range ou max amplitude ou bien user scale
        Mettre dans ce cas une cellule pour choisir l'instant
        Mettre un bouton pour chosir si on veut superposer l'image de départ avec les champs calculés par DIC
        Coder un calcul de résidu entre les champs de niveaux de gris par pixel reconstruits par DIC et les champs de niveau de gris de l'image d'origine 
        """
#####################################################################################
        #######################################################################
        # Code lines related to DIC - Code_Aster dialogue                     #
        #######################################################################
#####################################################################################


        code_aster_frame = ttk.LabelFrame(tab4, text='Preparation of Code_Aster test simulation')
        code_aster_frame.pack(expand=1, fill='x', padx=2, pady=2)

        # self.FRAME_for_icone=ttk.Frame(code_aster_frame, width=10, height =10)
        # self.FRAME_for_icone.place(in_=code_aster_frame, relx=1.0, x=-150, rely=0, y=5)

        # LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()

        """
        1- Avoir un générateur de scipt python pour mesh dans salome
        2- Avoir un système qui permet de choisir les lieux où on construira l'interpolation
        3- Mettre option sur le degré de la base d'interpolation
        4- Mettre option pour mise en données .comm:
            4-1- Modèle de comportement
            4-2- Imposer le type d'éléments en signalant
            4-3- Si non linéaire imposer le modèle de 'pilotage'
            4-4- Indiquer le type d'armature et leur emplacement et constuire générateur de maillage filaire
            4-5- Si non linéaire mettre options pour critère de convergence
            4-6- Si dynamique mettre option pour échantillonnage sortie, pas de temps de calcul
        """







#####################################################################################
        ###################################################################
        # Code lines related to the last tab of the GUI / About #
        ###################################################################
#####################################################################################


        # Frame pour user data
        self.about_frame = tk.LabelFrame(tab5, text='Credits')
        self.about_frame.grid(row=0,column=0)
        tab5.grid_columnconfigure(0, weight=1)
        tab5.grid_rowconfigure(0,weight=1)
        # self.about_frame.pack(
        #      expand=1,
        #         fill='both')
        # self.FRAME_for_icone=ttk.Frame(self.about_frame, width=10, height =10)
        # self.FRAME_for_icone.place(in_=self.about_frame, relx=0.5, rely=0, y=50)
        text_width=np.max([len(self.text_icone+' has been developed by'),len('Guillaume Hervé-Secourgeon'),len(' based on '+u"\N{GREEK SMALL LETTER MU}"+"DIC Python Class developed by S.N. Olufsen"),len('muDIC: An open-source toolkit for digital image correlation, S.N. Olufsen et al., SoftwareX Volume 11, January–June 2020, 100391')])

        LABEL=tk.Label(self.about_frame, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue'
                    #    ,width=text_width
                       ,anchor='center')

        # text_icone = ttk.Label(self.about_frame, text=
        #                                self.text_icone,
        #                                 width=text_width,justify='center',
                                    #    )
        LABEL.grid(row=0,column=0)

        text_version = tk.Label(self.about_frame, text=self.mudicgui_version,font=('Helvetica', 14,'bold','italic'),foreground='blue'
                    #    ,width=text_width
                       ,anchor='center')
        text_version.grid(row=1,column=0)
        # self.FRAME_for_credits=ttk.Frame(self.about_frame, width=text_width, height =200)
        # self.FRAME_for_credits.place(in_=self.about_frame, relx=0.5, rely=0, y=5)
        
        text_credits = tk.Label(self.about_frame, text=
                                       self.text_icone+' has been developed by\n Guillaume Hervé-Secourgeon\n based on '+u"\N{GREEK SMALL LETTER MU}"+"DIC Python Class developed by S.N. Olufsen \n muDIC: An open-source toolkit for digital image correlation, S.N. Olufsen et al., SoftwareX Volume 11, January–June 2020, 100391 \n https://www.sciencedirect.com/science/article/pii/S2352711019301967 \n MIT Licence \n (c) 2023 Copyright G. Hervé-Secourgeon"
                                        ,width=text_width,justify='center',anchor='center'
                                        ,font=('Helvetica', 10,'bold','italic')
                                       )
        text_credits.grid(row=2,column=0)
        #place(in_=self.about_frame,relx=0.5,rely=0.5)
        #x=np.int64(np.round(text_width)),







# Instanciation de l'application et exécution
app = muDIC_GUI()
app.run()
