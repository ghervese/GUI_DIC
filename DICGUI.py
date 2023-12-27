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
        self.root.title("User Interface for pre-processing images and DIC based on muDIC class")
        self.fig_photo_first = None
        self.source_selection_path = None

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
        
#     def select_output_images_folder(self):
#         """
#         This method is a dialog box to get the path accessing to the input images
#         """

# #        self.output_selection_path = filedialog.askdirectory(title='Select the output Directory for images cut on ROI')
# #        self.output_selection_path = os.path.join(self.output_selection_path, 'ROI_crop')
# #        os.makedirs(self.output_selection_path)

#         name = file_name.get() # took from the input
#         path = os.path.join(dir, name) # dir is a directory taken from the filedialog
#         default_folder = os.path.join(self.output_selection_path, 'ROI_crop')
#         if os.path.exists(default_folder) == True: #Only creates a new folder when non-existing
#             pass
#         else:
#             os.mkdir(default_folder)
#         file_directory = filedialog.askdirectory(title='Select the output Directory for images cut on ROI', initialdir = default_folder)



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



    def select_menu_num_images(self,event):
        self.num_images = self.list_combo_num_images.get()

    def select_menu_format_image(self,event):
        self.format_image = self.list_combo_format_image.get()


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


    def plot_ROI_on_fig(self):
        self.corner1 = np.array(self.corner1_entry.get())
        self.corner2 = np.array(self.corner2_entry.get())
        print(np.shape(self.corner1))
        print(np.shape(self.corner2))

          
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

    def create_output_folder(self):
        self.output_path = os.path.join(self.source_selection_path, 'ROI_cropped_images') # dir is a directory taken from the filedialog
        if os.path.exists(self.output_path) == True: #Only creates a new folder when non-existing
            pass
        else:
            os.mkdir(self.output_path)
        print(self.output_path)



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

        ##########################################################
        # Code relatif au premier onglet
        ##########################################################

        # Frame pour user data
        preprocessing_frame = ttk.LabelFrame(tab1, text='Pre-processing')
        preprocessing_frame.pack(expand=1, fill='both', padx=2, pady=2)

        chose_path_button = ttk.Button(preprocessing_frame, text="Select the folder location of the images", command=self.select_images_folder)
        chose_path_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        quit_button = ttk.Button(preprocessing_frame,text = "Quit", command = self.Close,width=5)
        quit_button.grid(row=0, column=6,columnspan=6, padx=5, pady=5)

        # Text area to define the prefix of the images
        object_label = ttk.Label(preprocessing_frame, text='Prefix of the images (replace the example):')
        object_label.grid(row=1, column=1, padx=5, pady=5)
        self.prefix_entry = ttk.Entry(preprocessing_frame)
        self.prefix_entry.insert(0, 'Gauche_Droite-')
        self.prefix_entry.grid(row=1, column=2, padx=5, pady=5)

        # Text area to define the prefix of the images
        object_label = ttk.Label(preprocessing_frame, text='Suffix of the images (replace the example):')
        object_label.grid(row=2, column=1, padx=5, pady=5)
        self.suffix_entry = ttk.Entry(preprocessing_frame)
        self.suffix_entry.insert(0, '_gauche')
        self.suffix_entry.grid(row=2, column=2, padx=5, pady=5)


        # Menu to select the numbering of the stack of images
        num_images = ttk.Label(preprocessing_frame, text='Numbering of the images (ex: if 00 selected then 00x):')
        num_images.grid(row=1, column=3, padx=5, pady=5)

        # list of the supported numbering types
        self.list_num_images = ['0','00','000','0000','00000','000000']
        # creation comboBox
        self.list_combo_num_images=ttk.Combobox(preprocessing_frame, values=self.list_num_images,width=6)
        default_num_image = 3
        self.list_combo_num_images.current(default_num_image)
        #ComboBox location
        self.list_combo_num_images.grid(row=1, column=4, padx=5, pady=5)
        self.list_combo_num_images.bind("<<ComboboxSelected>>",self.select_menu_num_images)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.num_images = self.list_num_images[default_num_image]

        # Menu to select the numbering of the stack of images
        format_image = ttk.Label(preprocessing_frame, text='Format of the images:')
        format_image.grid(row=2, column=3, padx=5, pady=5)

        # list of the supported numbering types
        self.list_format_image = ['.tif','.tiff','.png']
        # creation comboBox
        self.list_combo_format_image=ttk.Combobox(preprocessing_frame, values=self.list_format_image,width=5)
        default_format_image = 1
        self.list_combo_format_image.current(default_format_image)
        #Position de la ComboBox
        self.list_combo_format_image.grid(row=2, column=4, padx=5, pady=5)
        self.list_combo_format_image.bind("<<ComboboxSelected>>",self.select_menu_format_image)
        # Attribution of default value in case the user is satisfied with the proposed one
        self.format_image = self.list_format_image[default_format_image]

        import_first_image_button = ttk.Button(preprocessing_frame,text = "View first image", command = self.first_image_view)
        import_first_image_button.grid(row=2, column=5,columnspan=1, padx=5, pady=5)



        # Frame showing the first images of the stack
        self.preview_first_image_frame = ttk.LabelFrame(tab1, text='Field of view and ROI selection')
        self.preview_first_image_frame.pack(expand=0, fill='both', padx=2, pady=2)

        # Canvas pour afficher l'aperçu
        self.canvas_FOV_ROI = tk.Canvas(self.preview_first_image_frame)
#        self.canvas_FOV_ROI.pack(expand=1, fill='both')
        self.canvas_FOV_ROI.pack()

        ROI_frame = ttk.LabelFrame(tab1, text='Definition of the Region of Interest')
        ROI_frame.pack(expand=1, fill='both', padx=2, pady=2)

        select_ROI_button = ttk.Button(ROI_frame,text = "On-screen selection of the ROI", command = self.select_ROI_rectangle)
        select_ROI_button.grid(row=0, column=0, padx=5, pady=5)

        # show_ROI_button = ttk.Button(ROI_frame,text = "Apply ROI selection", command = self.plot_ROI_on_fig)
        # show_ROI_button.grid(row=0, column=1,columnspan=2, padx=5, pady=5)

        label_coord_corner1 = ttk.Label(ROI_frame, text='Coordinates in pixels of corner 1:')
        label_coord_corner1.grid(row=1, column=0, padx=5, pady=5)
        self.value_coord_corner1 = ttk.Label(ROI_frame, text='No coord. yet')
        self.value_coord_corner1.grid(row=1, column=1, padx=5, pady=5)
        
        # self.prefix_entry.insert(0, 'Gauche_Droite-')
        # self.prefix_entry.grid(row=1, column=3, padx=5, pady=5)

        label_coord_corner2 = ttk.Label(ROI_frame, text='Coordinates in pixels of corner 2:')
        label_coord_corner2.grid(row=1, column=2, padx=4, pady=5)
        self.value_coord_corner2 = ttk.Label(ROI_frame, text='No coord. yet')
        self.value_coord_corner2.grid(row=1, column=3, padx=5, pady=5)

        # def get_number_of_files():
        #     if self.source_selection_path is not None:
        #         var = tk.IntVar()
        #         #nb_files_max = len([entry for entry in os.listdir(self.source_selection_path)])
        #         scale = tk.Scale(ROI_frame, variable = var ,from_=1, to=self.nb_files_max,orient='horizontal',label='Number of files to crop',length=200)
        #         scale.grid(row=1,column=3, columnspan=2,padx=5, pady=5)
        #         self.nb_images_to_crop = var.get()

        # chose_path_button = ttk.Button(ROI_frame, text="Select the output folder location", command=lambda:[self.select_output_images_folder(),get_number_of_files()])
        # chose_path_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
   


        #prepare_resize_ROI_button = tk.Button(ROI_frame,text = "Generate default output directory", command=lambda:[self.create_output_folder, get_number_of_files()])
        prepare_resize_ROI_button1 = ttk.Button(ROI_frame,text = "Generate default output directory", command=self.create_output_folder)
        prepare_resize_ROI_button1.grid(row=2, column=0, padx=5, pady=5)

        # prepare_resize_ROI_button2 = ttk.Button(ROI_frame,text = "Number of files to be cropped", command=get_number_of_files())
        # prepare_resize_ROI_button2.grid(row=2, column=3,columnspan=2, padx=5, pady=5)

        # Text area to define the prefix of the images
        prepare_resize_ROI_files = ttk.Label(ROI_frame, text='Number of files to crop:')
        prepare_resize_ROI_files.grid(row=2, column=1, padx=5, pady=5)
        self.prepare_resize_ROI_files_entry = ttk.Entry(ROI_frame)
        self.prepare_resize_ROI_files_entry.insert(0, 2)
        self.prepare_resize_ROI_files_entry.grid(row=2, column=2, padx=5, pady=5)





        # Text area to define the prefix of the images
        prefix_ROI_files = ttk.Label(ROI_frame, text='User name for images:')
        prefix_ROI_files.grid(row=2, column=3, padx=5, pady=5)
        self.prefix_ROI_files_entry = ttk.Entry(ROI_frame)
        self.prefix_ROI_files_entry.insert(4, '_dynamic_beam_test_')
        self.prefix_ROI_files_entry.grid(row=2, column=4, padx=5, pady=5)



        resize_ROI_button = ttk.Button(ROI_frame,text = "Export images cropped on the ROI", command=self.crop_images_to_ROI)
        resize_ROI_button.grid(row=2, column=5,columnspan=2, padx=5, pady=5)


# Instanciation de l'application et exécution
app = muDIC_GUI()
app.run()
