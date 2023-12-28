import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        
    def FEM_select_images_folder(self):
        """
        This method is a dialog box to get the path accessing to the input images
        """

        self.FEM_source_selection_path = filedialog.askdirectory(title='Select the Directory hosting the images')
        self.FEM_nb_files_max = len([entry for entry in os.listdir(self.FEM_source_selection_path)])
        print('nbre de fichiers',self.FEM_nb_files_max)



    def line_select_callback(self,eclick, erelease):
        plt.gcf()
        plt.close('all')

        print('entre dans la selection')
        # x1, y1 = eclick.xdata, eclick.ydata
        # x2, y2 = erelease.xdata, erelease.ydata
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
        # x1, y1 = eclick.xdata, eclick.ydata
        # x2, y2 = erelease.xdata, erelease.ydata
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
        self.image_stack = dic.image_stack_from_folder(self.FEM_source_selection_path,file_type=self.FEM_format_image)



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

    def first_image_view(self):
        """
        This method loads the first image in the selected directory with the proper prefix, numbering and format.
        It converts it into a hypermatrix with pixels in row and columns and 4 digits corresponding to the Bayer matrix value
        If the image is in true gray level, this is not a hypermatrix but a "simple" matrix with 1 digit for each pixel location.
        """
        self.preview = mpimg.imread(self.source_selection_path +'/' +self.prefix_entry.get() + str(self.num_images) + '1' + self.format_image)

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

    def FEM_generate(self):
       
        xmax = max([np.int64(self.FEM_corner_1)[0],np.int64(self.FEM_corner_2)[0]])
        xmin = min([np.int64(self.FEM_corner_1)[0],np.int64(self.FEM_corner_2)[0]])
        ymax = max([np.int64(self.FEM_corner_1)[1],np.int64(self.FEM_corner_2)[1]])
        ymin = min([np.int64(self.FEM_corner_1)[1],np.int64(self.FEM_corner_2)[1]])

        nb_pix_par_elem = self.nb_pix_per_elem
        nb_elx = np.int64(np.floor((xmax-xmin)/nb_pix_par_elem))
        nb_ely = np.int64(np.floor((ymax-ymin)/nb_pix_par_elem))
        self.mesh = self.mesher.mesh(self.image_stack,Xc1=xmin,Xc2=xmax,Yc1=ymin,Yc2=ymax,n_ely=nb_ely,n_elx=nb_elx, GUI=False)

        self.FEM_ax.vlines(self.mesh.xnodes,min(self.mesh.ynodes),max(self.mesh.ynodes),color='r')
        self.FEM_ax.hlines(self.mesh.ynodes,min(self.mesh.xnodes),max(self.mesh.xnodes),color='r')

        self.FEM_fig_photo_first.get_tk_widget().destroy()

        self.FEM_fig_photo_first = FigureCanvasTkAgg(self.FEM_preview_selection, master=self.FEM_canvas_FOV_ROI)
        self.FEM_fig_photo_first.draw()
        self.FEM_fig_photo_first.get_tk_widget().pack()





    def FEM_first_image_view(self):
        """
        This method loads the first image in the selected directory with the proper prefix, numbering and format.
        It converts it into a hypermatrix with pixels in row and columns and 4 digits corresponding to the Bayer matrix value
        If the image is in true gray level, this is not a hypermatrix but a "simple" matrix with 1 digit for each pixel location.
        """
        self.FEM_preview = mpimg.imread(self.FEM_source_selection_path +'/' +self.FEM_prefix_entry.get() + str(self.FEM_num_images) + '1' + self.FEM_format_image)

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









    def crop_images_to_ROI(self):
        self.nb_images_to_crop = np.int64(self.prepare_resize_ROI_files_entry.get())
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
            image_entree = (self.source_selection_path +'/' +self.prefix_entry.get() +num_image + self.format_image)
            
            image = Image.open(image_entree) # On acquiert l'image
            image_input = np.asarray(image)

            x_min = np.int64(np.round(min(self.corner_1[1],self.corner_2[1])))
            x_max = np.int64(np.round(max(self.corner_1[1],self.corner_2[1])))
            y_min = np.int64(np.round(min(self.corner_1[0],self.corner_2[0])))
            y_max = np.int64(np.round(max(self.corner_1[0],self.corner_2[0])))

            image_cut_on_ROI = image_input[x_min:x_max,y_min:y_max] 

            Image.fromarray(image_cut_on_ROI).save(self.output_path+'/ROI' + self.prefix_ROI_files_entry.get() +num_image + self.format_image)
            if i == (self.nb_images_to_crop):
                print('Images have all been cropped !')

    def create_output_folder(self):
        self.output_path = os.path.join(self.source_selection_path, 'ROI_cropped_images') # dir is a directory taken from the filedialog
        if os.path.exists(self.output_path) == True: #Only creates a new folder when non-existing
            pass
        else:
            os.mkdir(self.output_path)
        print(self.output_path)

    def FEM_mesh_prop_gen(self):
        self.mesher = dic.Mesher(deg_e=1, deg_n=1,type=self.list_FEM_type)

    def DIC_generate_settings(self):
        self.DIC_settings = dic.DICInput(self.mesh,self.image_stack)
        self.DIC_settings.ref_update = tuple(self.first_image_for_DIC_entry.get())
        self.DIC_settings.max_nr_im = np.int64(self.first_image_for_DIC_entry.get())-np.int64(self.last_image_for_DIC_entry.get())+1
        self.DIC_settings.maxit = np.int64(self.iteration.get())
        self.DIC_settings.tom = np.float64(self.convergence.get())
        self.DIC_settings.interpolation_order = np.int64(self.order_interp.get())
        self.DIC_settings.store_internals=self.temp_store_internals

    def prepare_DIC_analysis(self):
        self.DIC_job = dic.DICAnalysis(self.DIC_settings)

    def run_DIC_analysis(self):
        self.DIC_results = self.DIC_job.run()

    def create_gui(self):
        """
        Creates the GUI

        This method uses the tkinter class to generate the different graphical elements of the GUI: buttons, text areas and tabs.
        It calls the different methods of the muDIC_GUI class
        """
        # Pre-processing of images
        tab_control = ttk.Notebook(self.root)
        tab1 = ttk.Frame(tab_control)
        tab_control.add(tab1, text='Pre-processing')
        tab_control.pack(expand=1, fill='both')

        # Definition of the ROI, creation of the mesh and calculation
        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text='Digital Image Correlation')
        tab_control.pack(expand=1, fill='both')

        # Post-processing of the different quantities of interest
        tab3 = ttk.Frame(tab_control)
        tab_control.add(tab3, text='Post-processing')
        tab_control.pack(expand=1, fill='both')

        tab4 = ttk.Frame(tab_control)
        tab_control.add(tab4, text='About')
        tab_control.pack(expand=1, fill='both')
#####################################################################################
        ###################################################################
        # Code lines related to the first tab of the GUI / Pre-processing #
        ###################################################################
#####################################################################################

        # Frame pour user data
        preprocessing_frame = ttk.LabelFrame(tab1, text='Selection of the experimental raw images')
        preprocessing_frame.pack(
             # expand=1,
                fill='both',
             # side='left',
              padx=2, pady=2)

        chose_path_button = ttk.Button(preprocessing_frame, text="Select the folder location of the images", command=self.select_images_folder)
        chose_path_button.grid(row=0, column=0, padx=2, pady=2)

        self.text_icone = u"\N{GREEK SMALL LETTER MU}"+'DIC'+u"\u1D33"+u"\u1D41"+u"\u1D35"
        # icone_frame = ttk.Frame(tab1)
        # icone_frame.place(in_=tab1, relx=1,rely=0,x=-100,y=-50)
        # icone = ttk.Label(icone_frame, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue')
        # icone.pack()

        self.FRAME_for_icone=ttk.Frame(preprocessing_frame, width=10, height =10)
        self.FRAME_for_icone.place(in_=preprocessing_frame, relx=1.0, x=-150, rely=0, y=5)

        LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()






        # self.FRAME_for_icone=ttk.Frame(FEM_frame, width=10, height =10)
        # self.FRAME_for_icone.place(in_=FEM_frame, relx=1.0, x=-100, rely=0, y=-10)

        # LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone).pack()


        quit_button = ttk.Button(preprocessing_frame,text = "Quit", command = self.Close,width=5)
        quit_button.place(in_=preprocessing_frame, relx=1.0, x=-130, rely=0, y=50)
        # quit_button.grid(row=1, column=4, padx=2, pady=2)

        # Text area to define the prefix of the images
        object_label = ttk.Label(preprocessing_frame, text='Prefix of the images (replace the example):')
        object_label.grid(row=1, column=0, padx=2, pady=2)
        self.prefix_entry = ttk.Entry(preprocessing_frame)
        self.prefix_entry.insert(0, 'Gauche_Droite-')
        self.prefix_entry.grid(row=1, column=1, padx=2, pady=2)

        # Text area to define the prefix of the images
        object_label = ttk.Label(preprocessing_frame, text='Suffix of the images (replace the example):')
        object_label.grid(row=2, column=0, padx=2, pady=2)
        self.suffix_entry = ttk.Entry(preprocessing_frame)
        self.suffix_entry.insert(0, '_gauche')
        self.suffix_entry.grid(row=2, column=1, padx=2, pady=2)

        text_num_images = 'Numbering format of the images \n (ex: if 00 selected then 00u, 0du ...):'
        # Menu to select the numbering of the stack of images
        num_images = ttk.Label(preprocessing_frame, text=text_num_images,anchor='e',justify='right',width=len('Numbering format of the images'))
        num_images.grid(row=1, column=2, padx=2, pady=2)

        # list of the supported numbering types
        self.list_num_images = ['0','00','000','0000','00000','000000']
        # creation comboBox
        self.list_combo_num_images=ttk.Combobox(preprocessing_frame, values=self.list_num_images,width=6)
        default_num_image = 3
        self.list_combo_num_images.current(default_num_image)
        #ComboBox location
        self.list_combo_num_images.grid(row=1, column=3, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.num_images = self.list_num_images[default_num_image]
        self.list_combo_num_images.bind("<<ComboboxSelected>>",self.select_menu_num_images)

        # Menu to select the numbering of the stack of images
        format_image = ttk.Label(preprocessing_frame, text='Type of the images:',anchor='e',width=len('Numbering format of the images'))
        format_image.grid(row=2, column=2, padx=2, pady=2)

        # list of the supported numbering types
        self.list_format_image = ['.tif','.tiff','.png']
        # creation comboBox
        self.list_combo_format_image=ttk.Combobox(preprocessing_frame, values=self.list_format_image,width=5)
        default_format_image = 1
        self.list_combo_format_image.current(default_format_image)
        #Position de la ComboBox
        self.list_combo_format_image.grid(row=2, column=3, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.format_image = self.list_format_image[default_format_image]
        self.list_combo_format_image.bind("<<ComboboxSelected>>",self.select_menu_format_image)

        import_first_image_button = ttk.Button(preprocessing_frame,text = "View first image", command = self.first_image_view)
        import_first_image_button.grid(row=2, column=4, padx=2, pady=2)

######################################################################################################
        # Frame showing the first images of the stack
######################################################################################################
        self.preview_first_image_frame = ttk.LabelFrame(tab1, text='Experimental field of view')
        self.preview_first_image_frame.pack(expand=0, fill='both', padx=2, pady=2)

        # Canvas pour afficher l'aperçu
        self.canvas_FOV_ROI = tk.Canvas(self.preview_first_image_frame)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.canvas_FOV_ROI.pack()


######################################################################################################
        # Frame for images cropping and generating of output cropped images
######################################################################################################
        

        ROI_frame = ttk.LabelFrame(tab1, text='Resizing the experimental images')
        ROI_frame.pack(expand=1, fill='both', padx=2, pady=2)

        select_ROI_button = ttk.Button(ROI_frame,text = "Resizing the image on screen", command = self.select_ROI_rectangle)
        select_ROI_button.grid(row=0, column=0, padx=2, pady=2)

        # show_ROI_button = ttk.Button(ROI_frame,text = "Apply ROI selection", command = self.plot_ROI_on_fig)
        # show_ROI_button.grid(row=0, column=1,columnspan=2, padx=2, pady=2)

        label_coord_corner1 = ttk.Label(ROI_frame, text='Coordinates in pixels of corner 1:')
        label_coord_corner1.grid(row=1, column=0, padx=2, pady=2)
        self.value_coord_corner1 = ttk.Label(ROI_frame, text='No coord. yet',foreground='blue')
        self.value_coord_corner1.grid(row=1, column=1, padx=2, pady=2)
        
        # self.prefix_entry.insert(0, 'Gauche_Droite-')
        # self.prefix_entry.grid(row=1, column=3, padx=2, pady=2)

        label_coord_corner2 = ttk.Label(ROI_frame, text='Coordinates in pixels of corner 2:')
        label_coord_corner2.grid(row=1, column=2, padx=4, pady=2)
        self.value_coord_corner2 = ttk.Label(ROI_frame, text='No coord. yet',foreground='blue')
        self.value_coord_corner2.grid(row=1, column=3, padx=2, pady=2)

   


        #prepare_resize_ROI_button = tk.Button(ROI_frame,text = "Generate default output directory", command=lambda:[self.create_output_folder, get_number_of_files()])
        prepare_resize_ROI_button1 = ttk.Button(ROI_frame,text = "Generate default output directory", command=self.create_output_folder)
        prepare_resize_ROI_button1.grid(row=2, column=0, padx=2, pady=2)

        # prepare_resize_ROI_button2 = ttk.Button(ROI_frame,text = "Number of files to be cropped", command=get_number_of_files())
        # prepare_resize_ROI_button2.grid(row=2, column=3,columnspan=2, padx=2, pady=2)

        # Text area to define the prefix of the images
        prepare_resize_ROI_files = ttk.Label(ROI_frame, text='Number of files to crop:')
        prepare_resize_ROI_files.grid(row=2, column=1, padx=2, pady=2)
        self.prepare_resize_ROI_files_entry = ttk.Entry(ROI_frame)
        self.prepare_resize_ROI_files_entry.insert(0, 2)
        self.prepare_resize_ROI_files_entry.grid(row=2, column=2, padx=2, pady=2)





        # Text area to define the prefix of the images
        prefix_ROI_files = ttk.Label(ROI_frame, text='User name for images:')
        prefix_ROI_files.grid(row=2, column=3, padx=2, pady=2)
        self.prefix_ROI_files_entry = ttk.Entry(ROI_frame)
        self.prefix_ROI_files_entry.insert(4, '_dynamic_beam_test_')
        self.prefix_ROI_files_entry.grid(row=2, column=4, padx=2, pady=2)



        resize_ROI_button = ttk.Button(ROI_frame,text = "Export cropped images", command=self.crop_images_to_ROI)
        resize_ROI_button.grid(row=3, column=0, padx=2, pady=2)

##################################################################################################
        ###############################################################################
        # Code lines related to the second tab of the GUI / Digital Image Correlation #
        ###############################################################################
##################################################################################################

        FEM_frame = ttk.LabelFrame(tab2, text='Finite Element properties of the ROI')
        FEM_frame.pack(expand=1, fill='both', padx=2, pady=2)

        self.FRAME_for_icone=ttk.Frame(FEM_frame, width=10, height =10)
        self.FRAME_for_icone.place(in_=FEM_frame, relx=1.0, x=-150, rely=0, y=5)

        LABEL=ttk.Label(self.FRAME_for_icone, text=self.text_icone,font=('Helvetica', 18,'bold','italic'),foreground='blue').pack()

        FEM_text_chose_path_button = "Select the folder location of the images"
        FEM_chose_path_button = ttk.Button(FEM_frame, text=FEM_text_chose_path_button, command=self.FEM_select_images_folder
                                    #    ,width=len(text_chose_path_button)
                                       )
        FEM_chose_path_button.grid(row=0, column=0, padx=2, pady=2)

        FEM_text_object_label = 'Prefix of the images (replace the example):'
        # Text area to define the prefix of the images
        FEM_object_label = ttk.Label(FEM_frame, text=FEM_text_object_label,width=len('Suffix of the images (replace the example):'),anchor='e'
                                #  width=len(text_object_label)
                                 )

        FEM_object_label.grid(row=1, column=0, padx=2, pady=2)
        self.FEM_prefix_entry = ttk.Entry(FEM_frame)
        self.FEM_prefix_entry.insert(0, 'ROI_dynamic_beam_test_')
        self.FEM_prefix_entry.grid(row=1, column=1, padx=2, pady=2)

        # Text area to define the prefix of the images
        FEM_text_object_label2 = 'Suffix of the images (replace the example):'
        FEM_object_label2 = ttk.Label(FEM_frame, text=FEM_text_object_label2,width=len('Suffix of the images (replace the example):'),anchor='e'
                                #   width=len(text_object_label2)
                                  )
        FEM_object_label2.grid(row=2, column=0, padx=2, pady=2)
        self.FEM_suffix_entry = ttk.Entry(FEM_frame)
        self.FEM_suffix_entry.insert(0, '')
        self.FEM_suffix_entry.grid(row=2, column=1, padx=2, pady=2)


        # Menu to select the numbering of the stack of images
        FEM_text_num_images = 'Numbering format of the images \n (ex: if 00 selected then 00u, 0du ...):'
        FEM_num_images = ttk.Label(FEM_frame, text=FEM_text_num_images,anchor='e',justify='right',width=len('Number of pixels per Finite Element side:')
                            #    width = len(text_num_images)
                               )
        FEM_num_images.grid(row=1, column=2, padx=2, pady=2)

        # list of the supported numbering types
        self.FEM_list_num_images = ['0','00','000','0000','00000','000000']
        # creation comboBox
        self.FEM_list_combo_num_images=ttk.Combobox(FEM_frame, values=self.FEM_list_num_images,width=6)
        FEM_default_num_image = 3
        self.FEM_list_combo_num_images.current(FEM_default_num_image)
        #ComboBox location
        self.FEM_list_combo_num_images.grid(row=1, column=3, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.FEM_num_images = self.FEM_list_num_images[default_num_image]
        self.FEM_list_combo_num_images.bind("<<ComboboxSelected>>",self.FEM_select_menu_num_images)

        # Menu to select the numbering of the stack of images
        FEM_text_format_image = 'Type of the images:'
        FEM_format_image = ttk.Label(FEM_frame, text=FEM_text_format_image,anchor='e',width=len('Number of pixels per Finite Element side:')
                                #  ,width=len(text_format_image)
                                 )
        FEM_format_image.grid(row=2, column=2, padx=2, pady=2)

        # list of the supported numbering types
        self.FEM_list_format_image = ['.tif','.tiff','.png']
        # creation comboBox
        self.FEM_list_combo_format_image=ttk.Combobox(FEM_frame, values=self.FEM_list_format_image,width=5)
        FEM_default_format_image = 1
        self.FEM_list_combo_format_image.current(default_format_image)
        #Position de la ComboBox
        self.FEM_list_combo_format_image.grid(row=2, column=3, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.FEM_format_image = self.FEM_list_format_image[FEM_default_format_image]
        self.FEM_list_combo_format_image.bind("<<ComboboxSelected>>",self.FEM_select_menu_format_image)

        FEM_text_import_first_image_button = "View first image"
        FEM_import_first_image_button = ttk.Button(FEM_frame,text = FEM_text_import_first_image_button, command = self.FEM_first_image_view
                                            #    , width=len(text_import_first_image_button)
                                               )
        FEM_import_first_image_button.grid(row=1, column=4, padx=2, pady=2)


        # Text area to define the prefix of the images
        FEM_nb_stack_images = '# of images to stack:'
        FEM_nb_stack_images = ttk.Label(FEM_frame, text=FEM_nb_stack_images,anchor='e',
                                #   width=len(text_object_label2)
                                        width=len('Number of pixels per Finite Element side:')
                                  )
        FEM_nb_stack_images.grid(row=0, column=2, padx=2, pady=2)
        self.FEM_nb_stack_images = ttk.Entry(FEM_frame,width=5)
        self.FEM_nb_stack_images.insert(0, 10)
        self.FEM_nb_stack_images.grid(row=0, column=3, padx=2, pady=2)



        FEM_text_chose_path_button = "Images stacking"
        FEM_chose_path_button = ttk.Button(FEM_frame, text=FEM_text_chose_path_button, command=self.image_stacking
                                    #    , width=len(text_chose_path_button)
                                       )
        FEM_chose_path_button.grid(row=2, column=4, padx=2, pady=2)


        # Menu to select the type of Finite Element
        text_FEM_type = 'Finite Element type:'
        FEM_type = ttk.Label(FEM_frame, text=text_FEM_type,anchor='e',width=len('Suffix of the images (replace the example):')
                            #  , width=len(text_FEM_type)
                             )
        FEM_type.grid(row=3, column=0, padx=2, pady=2)
       # list of the supported FEM - For the moment only Q4 are supported
        self.list_FEM_type = ['Q4','T3']
        # creation comboBox
        self.list_combo_FEM_type=ttk.Combobox(FEM_frame, values=self.list_FEM_type,width=5)
        default_FEM_type = 0
        self.list_combo_FEM_type.current(default_FEM_type)
        #Position de la ComboBox
        self.list_combo_FEM_type.grid(row=3, column=1, padx=2, pady=2)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.FEM_type = self.list_FEM_type[default_FEM_type]
        self.list_combo_FEM_type.bind("<<ComboboxSelected>>",self.select_menu_FEM_type)




        # Text area to define the number of pixels per Finite Element
        text_nb_pix_per_elem_label = 'Number of pixels per Finite Element side:'
        nb_pix_per_elem_label = ttk.Label(FEM_frame, text=text_nb_pix_per_elem_label,anchor='e'
                                           ,width=len(text_nb_pix_per_elem_label)
                                          )
        nb_pix_per_elem_label.grid(row=3, column=2, padx=2, pady=2)
        self.nb_pix_per_elem_entry = ttk.Entry(FEM_frame,width=6)
        self.nb_pix_per_elem_entry.insert(0, 20)
        self.nb_pix_per_elem_entry.grid(row=3, column=3, padx=2, pady=2)


        self.nb_pix_per_elem = np.int64(self.nb_pix_per_elem_entry.get())
        # nb_elx = np.int64(np.floor((xmax-xmin)/nb_pix_per_elem))
        # nb_ely = np.int64(np.floor((ymax-ymin)/nb_pix_per_elem))



        FEM_text_mesh_prop_button = "Generate FEM mesh properties"
        FEM_mesh_prop_button = ttk.Button(FEM_frame, text=FEM_text_mesh_prop_button, command=self.FEM_mesh_prop_gen
                                    #    , width=len(text_chose_path_button)
                                       )
        FEM_mesh_prop_button.grid(row=3, column=4,  padx=2, pady=2)








#########################################################################################
    # Frame for vizualizing the ROI on the cropped images
#########################################################################################
        self.FEM_preview_first_image_frame = ttk.LabelFrame(tab2, text='ROI selection of experimental image')
        self.FEM_preview_first_image_frame.pack(expand=0, fill='both', padx=2, pady=2)

        # Canvas pour afficher l'aperçu
        self.FEM_canvas_FOV_ROI = tk.Canvas(self.FEM_preview_first_image_frame)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.FEM_canvas_FOV_ROI.pack()

#############################################################################################
    # Frame for ROI construction and mesher generation
##############################################################################################

        FEM_ROI_frame = ttk.LabelFrame(tab2, text='Definition of the ROI')
        FEM_ROI_frame.pack(expand=1, fill='both', padx=2, pady=2)

        FEM_select_ROI_button = ttk.Button(FEM_ROI_frame,text = "Define rectangular ROI on screen", command = self.FEM_select_ROI_rectangle)
        FEM_select_ROI_button.grid(row=0, column=0, padx=2, pady=2)

        # show_ROI_button = ttk.Button(ROI_frame,text = "Apply ROI selection", command = self.plot_ROI_on_fig)
        # show_ROI_button.grid(row=0, column=1,columnspan=2, padx=2, pady=2)

        FEM_label_coord_corner1 = ttk.Label(FEM_ROI_frame, text='Coordinates in pixels of corner 1:')
        FEM_label_coord_corner1.grid(row=0, column=1, padx=2, pady=2)
        self.FEM_value_coord_corner1 = ttk.Label(FEM_ROI_frame, text='No coord. yet',foreground='blue')
        self.FEM_value_coord_corner1.grid(row=0, column=2, padx=2, pady=2)
        
        # self.prefix_entry.insert(0, 'Gauche_Droite-')
        # self.prefix_entry.grid(row=1, column=3, padx=2, pady=2)

        FEM_label_coord_corner2 = ttk.Label(FEM_ROI_frame, text='Coordinates in pixels of corner 2:')
        FEM_label_coord_corner2.grid(row=0, column=3, padx=4, pady=2)
        self.FEM_value_coord_corner2 = ttk.Label(FEM_ROI_frame, text='No coord. yet',foreground='blue')
        self.FEM_value_coord_corner2.grid(row=0, column=4, padx=2, pady=2)

        FEM_generate_button = ttk.Button(FEM_ROI_frame,text = "Create mesh", command = self.FEM_generate)
        FEM_generate_button.grid(row=0, column=6, padx=2, pady=2)



#########################################################################################

        DIC_solver_frame = ttk.LabelFrame(tab2, text='DIC solver')
        DIC_solver_frame.pack(expand=1, fill='both', padx=2, pady=2)

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



        FEM_generate_settings = ttk.Button(DIC_solver_frame,text = "Generate DIC settings", command = self.DIC_generate_settings)
        FEM_generate_settings.grid(row=2, column=0, padx=2, pady=2)


        iteration = ttk.Label(DIC_solver_frame, text='Maximum iterations:')
        iteration.grid(row=1, column=0, padx=2, pady=2)
        self.iteration = ttk.Entry(DIC_solver_frame,width=6)
        self.iteration.insert(0, 20)
        self.iteration.grid(row=1, column=1, padx=2, pady=2)

        convergence = ttk.Label(DIC_solver_frame, text='Convergence tolerance:')
        convergence.grid(row=1, column=2, padx=2, pady=2)
        self.convergence = ttk.Entry(DIC_solver_frame,width=6)
        self.convergence.insert(0, 1E-6)
        self.convergence.grid(row=1, column=3, padx=2, pady=2)

        order_interp = ttk.Label(DIC_solver_frame, text='Interpolation order:')
        order_interp.grid(row=1, column=4, padx=2, pady=2)
        self.order_interp = ttk.Entry(DIC_solver_frame,width=6)
        self.order_interp.insert(0, 20)
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

        prepare_DIC_analysis = ttk.Button(DIC_solver_frame,text = "Prepare analysis", command = self.prepare_DIC_analysis)
        prepare_DIC_analysis.grid(row=2, column=2, padx=2, pady=2)
        launch_DIC_analysis = ttk.Button(DIC_solver_frame,text = "Run DIC", command = self.run_DIC_analysis)
        launch_DIC_analysis.grid(row=2, column=3, padx=2, pady=2)


        """
        settings = dic.DICInput(mesh,image_stack)
        settings.max_nr_im = nbr_images_a_traiter
        # settings.ref_update = [5]
        settings.maxit = 20
        settings.tom = 1.e-6
        settings.interpolation_order = 4
        settings.store_internals = True
        # This setting defines the behaviour when convergence is not obtained
        settings.noconvergence = "ignore"
        dic_job = dic.DICAnalysis(settings)
        dic_results = dic_job.run()
        """


#####################################################################################
        ####################################################################
        # Code lines related to the third tab of the GUI / Post-processing #
        ####################################################################
#####################################################################################





#####################################################################################
        ###################################################################
        # Code lines related to the last tab of the GUI / About #
        ###################################################################
#####################################################################################

        # Frame pour user data
        self.about_frame = ttk.LabelFrame(tab4, text='Credits')
        self.about_frame.pack(
             expand=1,
                fill='both')
        text_width=np.max([len(self.text_icone+' has been developed by'),len('Guillaume Hervé-Secourgeon'),len(' based on '+u"\N{GREEK SMALL LETTER MU}"+"DIC Python Class developed by S.N. Olufsen")])
        text_credits = ttk.Label(self.about_frame, text=
                                       self.text_icone+' has been developed by\n Guillaume Hervé-Secourgeon\n based on '+u"\N{GREEK SMALL LETTER MU}"+"DIC Python Class developed by S.N. Olufsen \n MIT Licence \n (c) 2023 Copyright G. Hervé-Secourgeon"
                                        ,width=text_width,justify='center'
                                       )
        text_credits.pack(pady=100)
        #place(in_=self.about_frame,relx=0.5,rely=0.5)
        #x=np.int64(np.round(text_width)),

# Instanciation de l'application et exécution
app = muDIC_GUI()
app.run()
