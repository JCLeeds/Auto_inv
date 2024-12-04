import pylatex
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, Command, TextColor, Hyperref, Package, LongTable, MultiColumn
from pylatex.utils import italic, NoEscape, escape_latex
from pylatex.basic import NewPage
import os 
import glob
import numpy as np
def generateReport(GBIS_Area_path,GBIS_res_1,GBIS_res_2,ID):

    dirs = next(os.walk(GBIS_Area_path))[1]
    proc_dirs = [x for x in dirs if "processing_report_" in x]
    results_dir = [x for x in dirs if "INVERSION_Results" in x]
    NP1_res_dirs = next(os.walk(GBIS_res_1))[1]
    NP2_res_dirs = next(os.walk(GBIS_res_2))[1]
    extra_files = glob.glob(GBIS_Area_path+'/*.png')
    # print(extra_files)
    txt_files = glob.glob(GBIS_Area_path+'/*.txt')


    NP1_res_dirs_not_loc = [x for x in NP1_res_dirs if "location_run" not in x]
    # print(glob.glob(os.path.join(GBIS_res_2,NP1_res_dirs_not_loc[0]+'/Figures/*.png')))
    # print(glob.glob(os.path.join(GBIS_res_1,NP2_res_dirs[0]+'/Figures/*.png')))

    NP1_results_png = glob.glob(os.path.join(GBIS_res_1,NP2_res_dirs[0]+'/Figures/*.png'))
    NP2_results_png = glob.glob(os.path.join(GBIS_res_1,NP2_res_dirs[0]+'/Figures/*.png'))

    # geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    # doc = Document(geometry_options=geometry_options)
    # dirs = dirs[] 
    # print(proc_dirs)
    # print(results_dir)

    path_to_np1_dir = GBIS_Area_path + '/../' + ID + '_NP1'
    path_to_beachball_NP1 = GBIS_Area_path + '/' + ID +'_InSAR_beachball_NP1.png'
    # print(path_to_np1_dir+'/invert_*')
    inverts_NP1 = glob.glob(path_to_np1_dir+'/invert_*')
    # print(inverts_NP1)
    for invert in inverts_NP1:
        if 'location_run' in invert:
            # print(invert)
            pass 
        else:
            # print(invert)
            NP1_solution = invert
    # print(inverts_NP1)
    NP1_solution_figures = NP1_solution + '/Figures'
    NP1_txt_solution = glob.glob(NP1_solution +'/summary*')[0]
    with open(NP1_txt_solution,'r') as f:
        lines_NP1 = f.readlines()
    lines_NP1 = lines_NP1[7:len(lines_NP1)]
    starting_values_NP1 = [] 
    for line in lines_NP1:
        line_list = list(filter(bool,line.replace('\n','').replace('\t','').split(' ')))
        # print(line_list)
        starting_values_NP1.append(line_list[-1])
    # print((starting_values[8]))
    # print((starting_values[7]))
    rake_start = np.rad2deg(np.arctan2(-float(starting_values_NP1[8]),-float(starting_values_NP1[7])))
    rake_start =  round(rake_start, 6)
    starting_values_NP1 = [rake_start] + starting_values_NP1[0:len(starting_values_NP1)-1]




    path_to_np2_dir = GBIS_Area_path + '/../' + ID + '_NP2'
    path_to_beachball_NP2 = GBIS_Area_path + '/' + ID +'_InSAR_beachball_NP2.png'
    print(path_to_np2_dir+'/invert_*')
    inverts_NP2 = glob.glob(path_to_np2_dir+'/invert_*')
    for invert in inverts_NP2:
        if 'location_run' in invert:
            pass 
        else:
            NP2_solution = invert
    # print(inverts_NP2)
    NP2_txt_solution = glob.glob(NP2_solution +'/summary*')[0]
    with open(NP2_txt_solution,'r') as f:
        lines_NP2 = f.readlines()
    lines_NP2 = lines_NP2[7:len(lines_NP2)]
    starting_values_NP2 = [] 
    for line in lines_NP2:
        line_list = list(filter(bool,line.replace('\n','').replace('\t','').split(' ')))
        # print(line_list)
        starting_values_NP2.append(line_list[-1])
    # print(float(-starting_values[-8]))
    # print(float(-starting_values[-7]))
    rake_start = np.rad2deg(np.arctan2(-float(starting_values_NP2[8]),-float(starting_values_NP2[7])))
    rake_start =  round(rake_start, 6)
    starting_values_NP2 = [rake_start] + starting_values_NP2[0:len(starting_values_NP2)-1]




    frames = []
    for ii in range(len(proc_dirs)):
        frames.append(proc_dirs[ii].split('_processing_report')[0])

    inv_date_NP1 = [] 
    inv_date_NP2 = []
    pro_png_NP1 = [] 
    pro_png_NP2 = []
    for frame in frames:
            # geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
            # doc = Document(geometry_options=geometry_options)
        
            # doc.generate_pdf('full', clean_tex=False)
            top_dir_NP1_inv = os.path.join(GBIS_Area_path,frame + '_INVERSION_Results_NP1')
            top_dir_NP2_inv = os.path.join(GBIS_Area_path,frame + '_INVERSION_Results_NP2')
            top_dir_NP1_pro = os.path.join(GBIS_Area_path,frame + '_processing_report_NP1')
            top_dir_NP2_pro = os.path.join(GBIS_Area_path,frame + '_processing_report_NP2')
            dates = next(os.walk(top_dir_NP1_inv))[1]
            pro_png_NP1.append(glob.glob(os.path.join(GBIS_Area_path,frame + '_processing_report_NP1/'+'*.png')))
            pro_png_NP2.append(glob.glob(os.path.join(GBIS_Area_path,frame + '_processing_report_NP2/'+'*.png')))
            inv_date_NP1 = [] 
            inv_date_NP2 = []
            for date in dates:
                inv_date_NP1.append(glob.glob(os.path.join(top_dir_NP1_inv,date+'/*.png')))
                inv_date_NP2.append(glob.glob(os.path.join(top_dir_NP2_inv,date+'/*.png')))

    path_to_event_file = os.path.join(GBIS_Area_path,ID+'.txt')
    path_to_seismic_png = os.path.join(GBIS_Area_path,ID+'_Seismic_beachball.png')
    path_to_location_png = os.path.join(GBIS_Area_path,'location_and_active_frames_plot.png')
    path_to_active_frame_info = os.path.join(GBIS_Area_path,'frames_sent_to_gbis.txt')
    path_to_ifgms_used = os.path.join(GBIS_Area_path,'ifgms_used_in_inversion.txt')
    path_to_auto_inv = os.getcwd()
    with open(path_to_event_file,'r') as file:
            params = file.readlines()
  
    name = params[0].split('=')[-1]
    time = params[1].split('=')[-1]
    latitude = float(params[2].split('=')[-1])
    longitude = float(params[3].split('=')[-1])
    magnitude = float(params[4].split('=')[-1])
    magnitude_type = params[5].split('=')[-1]
    moment = float(params[6].split('=')[-1])
    depth = float(params[7].split('=')[-1])
    catalog = params[8].split('=')[-1]
    strike1 = float(params[9].split('=')[-1])
    dip1 = float(params[10].split('=')[-1])
    rake1 = float(params[11].split('=')[-1])
    strike2 = float(params[12].split('=')[-1])
    dip2 = float(params[13].split('=')[-1])
    rake2 = float(params[14].split('=')[-1])


    geometry_options = {"tmargin": "1cm", "lmargin": "1cm", 'rmargin':'1cm', 'bmargin':'1cm'}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('hyperref'))
    doc.preamble.append(NoEscape(r'\usepackage[section]{placeins}'))
    # doc = Document()


    # doc.preamble.append(NoEscape(r'\usepackage{vmargin}'))
    doc.preamble.append(NoEscape(r'\usepackage{titlesec}'))
    doc.preamble.append(NoEscape(r'\titleformat{\section}[block]'))
    doc.preamble.append(NoEscape(r'  {\fontsize{18}{23}\bfseries}'))
    doc.preamble.append(NoEscape(r'  {\thesection}'))
    doc.preamble.append(NoEscape(r'  {1em}'))
    doc.preamble.append(NoEscape(r'  {}'))
    doc.preamble.append(NoEscape(r'\titleformat{\subsection}[hang]'))
    doc.preamble.append(NoEscape(r'  {\fontsize{18}{23}\bfseries}'))
    doc.preamble.append(NoEscape(r'  {\thesubsection}'))
    doc.preamble.append(NoEscape(r'  {1em}'))
    doc.preamble.append(NoEscape(r'  {}'))
    doc.preamble.append(NoEscape(r'\titleformat{\subsubsection}[hang]'))
    doc.preamble.append(NoEscape(r'  {\fontsize{18}{23}\bfseries}'))
    doc.preamble.append(NoEscape(r'  {\thesubsubsection}'))
    doc.preamble.append(NoEscape(r'  {1em}'))
    doc.preamble.append(NoEscape(r'  {}'))

    # doc.preamble.append(NoEscape(r'\setmarginsrb{1.0 cm}{.75 cm}{1.0 cm}{.75 cm}{1 cm}{.75 cm}{1 cm}{.75 cm}'))
    doc.preamble.append(NoEscape(r'\title{\begin{center}'))
    doc.preamble.append(NoEscape(r'\fontsize{25pt}{25pt}\selectfont\bfseries InSAR Source Parameter Inversion Report for ' + ID ))
    doc.preamble.append(NoEscape(r'\end{center}}	'))
    doc.preamble.append(NoEscape(r'\author{J. Condon}	'))
    doc.preamble.append(NoEscape(r'\date{\today}	'))
    doc.preamble.append(NoEscape(r'\makeatletter'))
    doc.preamble.append(NoEscape(r'\let\thetitle\@title'))
    doc.preamble.append(NoEscape(r'\let\theauthor\@author'))
    doc.preamble.append(NoEscape(r'\let\thedate\@date  '))

    
    
    # doc.preamble.append(NoEscape(r'\usepackage{eso-pic}'))
    # doc.preamble.append(Command('title', f"Inversion Report for {ID}"))
    # doc.preamble.append(Command('author', "GABIS"))
    # doc.preamble.append(Command('date', NoEscape(r'\vspace{-3ex}'))) #didn't want a date
    # doc.append(NoEscape(r'\maketitle'))

    doc.append(NoEscape(r'\begin{titlepage}'))
    doc.append(NoEscape(r'\centering'))
    doc.append(NoEscape(r'\vspace{0.5 cm}'))
    doc.append(NoEscape(r'\centering'))
    doc.append(NoEscape(r'\includegraphics[scale = 0.1]{../Logo.jpg}\\[1.0 cm]'))
    doc.append(NoEscape(r'\textsc{\LARGE University of Leeds}\\[2.0 cm]'))
    doc.append(NoEscape(r'\textsc{\Large School of Earth and Environment}\\[0.5 cm]'))
    doc.append(NoEscape(r'\textsc{\large GABIS Inversion Report. }\\[0.5 cm]'))
    doc.append(NoEscape(r'\rule{\linewidth}{0.2 mm} \\[0.4 cm]'))
    doc.append(NoEscape(r'{\huge \bfseries \thetitle}'))
    doc.append(NoEscape(r'\rule{\linewidth}{0.2 mm} \\[1.5 cm] '))
    doc.append(NoEscape(r'\begin{minipage}{0.4\textwidth}'))
    doc.append(NoEscape(r'\begin{center} \large'))
    doc.append(NoEscape(r'\emph{Author:}\\'))
    doc.append(NoEscape(r'	J. Condon'))
    doc.append(NoEscape(r'\end{center}'))
    doc.append(NoEscape(r'\end{minipage}~\\[2 cm]'))
    # doc.preamble.append(NoEscape(r'\begin{minipage}{0.4\textwidth}'))
    # doc.preamble.append(NoEscape(r'\begin{center}'))
    # doc.preamble.append(NoEscape(r'\emph{auto_generted:} \\'))
    # doc.preamble.append(NoEscape(r'Geodetic Automated Baysian Inversion Software (GABIS)'))
    # doc.preamble.append(NoEscape(r'\end{center}'))
    # doc.preamble.append(NoEscape(r'\end{minipage}\\[2 cm] '))
    doc.append(NoEscape(r'\end{titlepage}'))


    with doc.create(Section('Event Information')):
        #  {"tmargin": "1cm", "lmargin": "1cm", 'rmargin':'1cm', 'bmargin':'1cm'}
        doc.geometry_options = Command( {"tmargin": "1cm", "lmargin": "1cm", 'rmargin':'1cm', 'bmargin':'1cm'})
        doc.append('USGS ID: ' + name )
        doc.append('Event Time: ' + time)
        doc.append('Latitude: ' + str(latitude)+'\n')
        doc.append('Longitude: ' + str(longitude)+'\n')
        doc.append('Magnitude Type: ' + str(magnitude_type)+'\n')
        doc.append('Magnitude: ' + str(magnitude)+'\n')
        doc.append('Moment: ' + str(moment)+'Nm' +'\n')
        doc.append('Depth: ' + str(depth/1000)+'km'+'\n')
        doc.append('Strike NP1: ' + str(strike1)+'° ')
        doc.append('Dip NP1: ' + str(dip1)+'° ')
        doc.append('Rake NP1: ' + str(rake1)+'°'+ '\n')
        doc.append('Strike NP2: ' + str(strike2) + '° ')
        doc.append('Dip NP2: ' + str(dip2)+ '° ')
        doc.append('Rake NP2: ' + str(rake2)+'° ')

        with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["MODEL PARAM", "OPTIMAL NP1", "MEAN NP1",'MEDIAN NP1','2.5%','97.5%','STARTING NP1'])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row((MultiColumn(7, align='r'
                               ),))
            data_table.add_hline()
            data_table.end_table_footer()
            data_table.add_hline()
            data_table.add_row((MultiColumn(7, align='r'),))
            data_table.add_hline()
            data_table.end_table_last_footer()
            for ii,line in enumerate(lines_NP1):
                row_list =  list(filter(bool,line.replace('\n','').replace('\t','').split(' ')))
                row_list[-1] = starting_values_NP1[ii]
                if row_list[0] == 'Dip':
                    row_list[1:len(row_list)] = np.array(row_list[1:len(row_list)],dtype=float) * -1
                elif row_list[0] == 'Strike':
                    row_list[1:len(row_list)] = np.array(row_list[1:len(row_list)],dtype=float) + 180 % 360
                array = np.array(row_list[1:len(row_list)], dtype=float)
                row_list[1:len(row_list)] = np.round(array,decimals=2)
                print(row_list)
                data_table.add_row(row_list)
               

        
        with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["MODEL PARAM", "OPTIMAL NP2", "MEAN NP2",'MEDIAN NP2','2.5%','97.5%','STARTING NP2'])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row((MultiColumn(7, align='r',
                                data='NP2'),))
            data_table.add_hline()
            data_table.end_table_footer()
            data_table.add_hline()
            data_table.add_row((MultiColumn(7, align='r',
                                data='NP2'),))
            data_table.add_hline()
            data_table.end_table_last_footer()
            
            for ii,line in enumerate(lines_NP2):
                row_list =  list(filter(bool,line.replace('\n','').replace('\t','').split(' ')))
                row_list[-1] = starting_values_NP2[ii]

                if row_list[0] == 'Dip':
                    row_list[1:len(row_list)] = np.array(row_list[1:len(row_list)],dtype=float) * -1
                elif row_list[0] == 'Strike':
                    row_list[1:len(row_list)] = np.array(row_list[1:len(row_list)],dtype=float) + 180 % 360
                
                array = np.array(row_list[1:len(row_list)], dtype=float)
                row_list[1:len(row_list)] = np.round(array,decimals=2)
                print(row_list)
                # print(row_list)
                data_table.add_row(row_list)

        with doc.create(Figure(position='h!')) as pic:
            pic.add_image(path_to_seismic_png)
            pic.add_caption('USGS Focal Mechanism')
        
        with doc.create(Figure(position='h!')) as pic:
            pic.add_image(path_to_beachball_NP1)
            pic.add_caption('InSAR NP1 Solution')

        with doc.create(Figure(position='h!')) as pic:
            pic.add_image(path_to_beachball_NP2)
            pic.add_caption('InSAR NP2 Solution')


        with doc.create(Figure(position='h!')) as pic:
            pic.add_image(path_to_location_png)
            pic.add_caption('Location of event with LiCS frames activated and processed, only frames that pass quality check are used in final inversion')

    with open(path_to_active_frame_info,'r') as file:
            mat_files = file.readlines()
    active_frames = [] 
    for mat_file in mat_files:
        active_frames.append(mat_file.split('_floatml')[0].split('EOC_')[-1])
        
    # print(active_frames)
    # doc.append(NewPage())
    # GEOC_166D_06139_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_down_sampled
    try:
        with open(path_to_ifgms_used,'r') as file:
                ifgms_full_path = file.readlines()
        active_ifgms = []
        for path_ifgms in ifgms_full_path:
            active_ifgms.append(path_ifgms.split('_QAed_')[-1].split('.')[0])     
    except:
        ifgms_full_path = [] 
        active_ifgms = []
   
    # path_to_np1_dir = GBIS_Area_path + '/../' + ID + '_NP1'
    # path_to_beachball_NP1 = GBIS_Area_path + '/' + ID +'_InSAR_beachball_NP1.png'
    # # print(path_to_np1_dir+'/invert_*')
    # inverts_NP1 = glob.glob(path_to_np1_dir+'/invert_*')
    # # print(inverts_NP1)
    # for invert in inverts_NP1:
    #     if 'location_run' in invert:
    #         # print(invert)
    #         pass 
    #     else:
    #         # print(invert)
    #         NP1_solution = invert
    # # print(inverts_NP1)
    # NP1_solution_figures = NP1_solution + '/Figures'
    # NP1_txt_solution = glob.glob(NP1_solution +'/summary*')[0]
    # with open(NP1_txt_solution,'r') as f:
    #     lines = f.readlines()
    # lines = lines[7:len(lines)]
    # starting_values = [] 
    # for line in lines:
    #     line_list = list(filter(bool,line.replace('\n','').replace('\t','').split(' ')))
    #     # print(line_list)
    #     starting_values.append(line_list[-1])
    # # print((starting_values[8]))
    # # print((starting_values[7]))
    # rake_start = np.rad2deg(np.arctan2(-float(starting_values[8]),-float(starting_values[7])))
    # rake_start =  round(rake_start, 6)
    # starting_values = [rake_start] + starting_values[0:len(starting_values)-1]


        
    data_model_res = glob.glob(NP1_solution_figures + '/*res_los_modlos*.png')
    # doc.append(NewPage(1))
    # doc.preamble.append(NoEscape(r'\clearpage'))
    with doc.create(Section('Inversion Results NP1')):
        path_to_convergence = NP1_solution_figures + '/convergence_fault0.png'
        path_to_jointprobs = NP1_solution_figures + '/probability-density-distributions_fault0.png'
        path_to_PDF = glob.glob(NP1_solution_figures+'/pdf_fault*')[0]

        with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["MODEL PARAM", "OPTIMAL NP1", "MEAN NP1",'MEDIAN NP1','2.5%','97.5%','STARTING NP1'])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row((MultiColumn(7, align='r'
                               ),))
            data_table.add_hline()
            data_table.end_table_footer()
            data_table.add_hline()
            data_table.add_row((MultiColumn(7, align='r'),))
            data_table.add_hline()
            data_table.end_table_last_footer()
            for ii,line in enumerate(lines_NP1):
                row_list =  list(filter(bool,line.replace('\n','').replace('\t','').split(' ')))
                row_list[-1] = starting_values_NP1[ii]
                if row_list[0] == 'Dip':
                    row_list[1:len(row_list)] = np.array(row_list[1:len(row_list)],dtype=float) * -1
                    
                elif row_list[0] == 'Strike':
                    row_list[1:len(row_list)] = np.array(row_list[1:len(row_list)],dtype=float) + 180 % 360
                
                array = np.array(row_list[1:len(row_list)], dtype=float)
                row_list[1:len(row_list)] = np.round(array,decimals=2)
                # print(row_list)
                data_table.add_row(row_list)

            
        # print(path_to_PDF)
        with doc.create(Figure(position='h!')) as pic:
                        pic.add_image(path_to_jointprobs, width='350px')
                        pic.add_caption('Joint Probabilities NP1 ')
        with doc.create(Figure(position='h!')) as pic:
                        pic.add_image(path_to_convergence, width='350px')
                        pic.add_caption('Convergence NP1')
        with doc.create(Figure(position='h!')) as pic:
                        pic.add_image(path_to_PDF, width='350px')
                        pic.add_caption('Probability distibution functions NP1')
        with doc.create(Figure(position='h!')) as pic:
                        pic.add_image(path_to_beachball_NP1, width='100px')
                        pic.add_caption('Focal mechanism plot using NP1 solution')

        doc.append(NoEscape(r'\FloatBarrier'))


        link_to_forward_NP1 = []
        link_to_inversion_NP1 = []
        link_to_location_NP1 = []
        link_to_semi_varigoram = []
        downsamp_mosaic = []
        list_of_frames = [] 
        list_of_dates = [] 

        for frame in active_frames:
            frame_dir_inv = os.path.join(GBIS_Area_path,'GEOC_' + frame + '_INVERSION_Results_NP1')
            frame_dir_proc = os.path.join(GBIS_Area_path,'GEOC_' + frame + '_processing_report_NP1')
            dates = [ f.path for f in os.scandir(frame_dir_inv) if f.is_dir() ]
            downsamp_mosaic.append(frame_dir_proc + '/All_ifgms_easy_look_up_downsamp.png')
            
            # if os.path.isfile(downsamp_mosaic):
            #     with doc.create(Figure(position='h!')) as pic:
            #                 pic.add_image(downsamp_mosaic, width='350px')
            #                 pic.add_caption('Easy look up of all dates used in inversion.' )
            # doc.append(NoEscape(r'\FloatBarrier'))
            # with doc.create(Subsection(' USGS Forward Models for All Dates in ' + str(frame)+ ' NP1')):
            for date in dates:
                    print(date)
                    print(active_ifgms)
                    if date.split('/')[-1] in active_ifgms:
                        print('__________________HERE_________________')
                        link_to_forward_NP1.append(frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_1.png')
                        link_to_inversion_NP1.append(frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png')
                        link_to_location_NP1.append(frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png')
                        link_to_semi_varigoram.append(frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png')
                        list_of_frames.append(frame) 
                        list_of_dates.append(date)


        

        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('Model Generated from Inversion NP1')):
            for ii, path in enumerate(link_to_inversion_NP1):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Inversion results for NP1 for frame ' + str(list_of_frames[ii]) + ' on dates ' + list_of_dates[ii].split('/')[-1])

        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('Model Generated from USGS NP1')):
            for  ii, path in enumerate(link_to_forward_NP1):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Inversion results for NP1 for frame ' + str(list_of_frames[ii]) + ' on dates '  + list_of_dates[ii].split('/')[-1])


        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('Fault Plane Generated from Inversion NP1')):
            for  ii, path in enumerate(link_to_location_NP1):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Inversion results for NP1 for frame ' + str(list_of_frames[ii]) + ' on dates '  + list_of_dates[ii].split('/')[-1])


        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('Ifgms Used in Inversion')):
            for ii, path in enumerate(downsamp_mosaic):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Ifgms Used in Inversion from frame ' + str(active_frames[ii]) + ' on dates ' + list_of_dates[ii].split('/')[-1])

        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('SemiVariograms of Data used in Inversion')):
            for  ii, path in enumerate(link_to_semi_varigoram):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Inversion results for NP1 for frame ' + str(list_of_frames[ii]) + ' on dates '  + list_of_dates[ii].split('/')[-1])






            #             with doc.create(Figure(position='ht!')) as pic:
            #                 pic.add_image(link_to_forward_NP1, width='350px')
            #                 print(link_to_forward_NP1)
            #                 pic.add_caption('USGS forward model nodal plane one for dates ' + date.split('/')[-1] )
            # doc.append(NoEscape(r'\FloatBarrier'))
            # with doc.create(Subsection('Model Generated from Inversion NP1')):
            #     for date in dates:
            #         if date.split('/')[-1] in active_ifgms:
            #             print('__________________HERE_________________')
            #             link_to_forward_NP1 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_1.png'
            #             link_to_inversion_NP1 =  frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png'
            #             link_to_location_NP1 = frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png'
            #             link_to_semi_varigoram = frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png'


            #             with doc.create(Figure(position='ht!')) as pic:
            #                 pic.add_image(link_to_inversion_NP1, width='350px')
            #                 pic.add_caption('Inversion results for NP1 for dates ' + date.split('/')[-1])
            # doc.append(NoEscape(r'\FloatBarrier'))
            # with doc.create(Subsection('Fault Plane Generated from Inversion NP1')):  
            #     for date in dates:
            #         if date.split('/')[-1] in active_ifgms:
            #             print('__________________HERE_________________')
            #             link_to_forward_NP1 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_1.png'
            #             link_to_inversion_NP1 =  frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png'
            #             link_to_location_NP1 = frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png'
            #             link_to_semi_varigoram = frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png'

            #             if os.path.isfile(link_to_location_NP1):
            #                 with doc.create(Figure(position='ht!')) as pic:
            #                     pic.add_image(link_to_location_NP1, width='350px')
            #                     pic.add_caption('Location results for NP1 for dates ' + date.split('/')[-1])
            # doc.append(NoEscape(r'\FloatBarrier'))
            # with doc.create(Subsection('SemiVariograms for NP1')):  
            #     for date in dates:
            #         if date.split('/')[-1] in active_ifgms:
            #             print('__________________HERE_________________')
            #             link_to_forward_NP1 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_1.png'
            #             link_to_inversion_NP1 =  frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png'
            #             link_to_location_NP1 = frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png'
            #             link_to_semi_varigoram = frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png'
            #             if os.path.isfile(link_to_semi_varigoram):
            #                 with doc.create(Figure(position='ht!')) as pic:
            #                     pic.add_image(link_to_semi_varigoram, width='350px')
            #                     pic.add_caption('Exponential fit Semivariogram for data with signal mask' + date.split('/')[-1] )


        

    path_to_np2_dir = GBIS_Area_path + '/../' + ID + '_NP2'
    path_to_beachball_NP2 = GBIS_Area_path + '/' + ID +'_InSAR_beachball_NP2.png'
    print(path_to_np2_dir+'/invert_*')
    inverts_NP2 = glob.glob(path_to_np2_dir+'/invert_*')
    for invert in inverts_NP2:
        if 'location_run' in invert:
            pass 
        else:
            NP2_solution = invert
    # print(inverts_NP2)
    NP2_txt_solution = glob.glob(NP2_solution +'/summary*')[0]
    with open(NP2_txt_solution,'r') as f:
        lines_NP2 = f.readlines()
    lines_NP2 = lines_NP2[7:len(lines_NP2)]
    starting_values_NP2 = [] 
    for line in lines_NP2:
        line_list = list(filter(bool,line.replace('\n','').replace('\t','').split(' ')))
        # print(line_list)
        starting_values_NP2.append(line_list[-1])
    # print(float(-starting_values[-8]))
    # print(float(-starting_values[-7]))
    rake_start = np.rad2deg(np.arctan2(-float(starting_values_NP2[8]),-float(starting_values_NP2[7])))
    rake_start =  round(rake_start, 6)
    starting_values_NP2 = [rake_start] + starting_values_NP2[0:len(starting_values_NP2)-1]


    NP2_solution_figures = NP2_solution + '/Figures'
    data_model_res = glob.glob(NP2_solution_figures + '/*res_los_modlos*.png')
    # doc.append(NewPage(1))
    # doc.preamble.append(NoEscape(r'\clearpage'))
    doc.append(NoEscape(r'\FloatBarrier'))
    with doc.create(Section('Inversion Results NP2')):
        path_to_convergence = NP2_solution_figures + '/convergence_fault0.png'
        path_to_jointprobs = NP2_solution_figures + '/probability-density-distributions_fault0.png'
        path_to_PDF = glob.glob(NP2_solution_figures+'/pdf_fault0.png')[0]
        # print(path_to_PDF)
        with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["MODEL PARAM", "OPTIMAL NP2", "MEAN NP2",'MEDIAN NP2','2.5%','97.5%','STARTING NP2'])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row((MultiColumn(7, align='r',
                                data='Continued on Next Page'),))
            data_table.add_hline()
            data_table.end_table_footer()
            data_table.add_hline()
            data_table.add_row((MultiColumn(7, align='r',
                                data='Not Continued on Next Page'),))
            data_table.add_hline()
            data_table.end_table_last_footer()
            
            for ii,line in enumerate(lines_NP2):
                row_list =  list(filter(bool,line.replace('\n','').replace('\t','').split(' ')))
                row_list[-1] = starting_values_NP2[ii]
                if row_list[0] == 'Dip':
                    row_list[1:len(row_list)] = np.array(row_list[1:len(row_list)],dtype=float) * -1
                elif row_list[0] == 'Strike':
                    row_list[1:len(row_list)] = np.array(row_list[1:len(row_list)],dtype=float) + 180 % 360

                
                array = np.array(row_list[1:len(row_list)], dtype=float)
                row_list[1:len(row_list)] = np.round(array,decimals=2)
                # print(row_list)
                data_table.add_row(row_list)

        with doc.create(Figure(position='h!')) as pic:
                        pic.add_image(path_to_jointprobs, width='350px')
                        pic.add_caption('Joint probabilities NP2 ')
        with doc.create(Figure(position='h!')) as pic:
                        pic.add_image(path_to_convergence, width='350px')
                        pic.add_caption('Convergence NP2')
        with doc.create(Figure(position='h!')) as pic:
                        pic.add_image(path_to_PDF, width='350px')
                        pic.add_caption('Probability distibution functions NP2')

        with doc.create(Figure(position='h!')) as pic:
                        pic.add_image(path_to_beachball_NP2, width='100px')
                        pic.add_caption('Focal mechanism plot using NP2 solution')

      

        doc.append(NoEscape(r'\FloatBarrier'))
        link_to_forward_NP2 = []
        link_to_inversion_NP2 = []
        link_to_location_NP2 = []
        link_to_semi_varigoram_2 = []
        downsamp_mosaic_2 = []
        list_of_frames = []
        list_of_dates = [] 

        for frame in active_frames:
            frame_dir_inv = os.path.join(GBIS_Area_path,'GEOC_' + frame + '_INVERSION_Results_NP2')
            frame_dir_proc = os.path.join(GBIS_Area_path,'GEOC_' + frame + '_processing_report_NP2')
            dates = [ f.path for f in os.scandir(frame_dir_inv) if f.is_dir() ]
            downsamp_mosaic_2.append(frame_dir_proc + '/All_ifgms_easy_look_up_downsamp.png')
            
            # if os.path.isfile(downsamp_mosaic):
            #     with doc.create(Figure(position='h!')) as pic:
            #                 pic.add_image(downsamp_mosaic, width='350px')
            #                 pic.add_caption('Easy look up of all dates used in inversion.' )
            # doc.append(NoEscape(r'\FloatBarrier'))
            # with doc.create(Subsection(' USGS Forward Models for All Dates in ' + str(frame)+ ' NP1')):
            for date in dates:
                    print(date)
                    print(active_ifgms)
                    if date.split('/')[-1] in active_ifgms:
                        print('__________________HERE_________________')
                        link_to_forward_NP2.append(frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_2.png')
                        link_to_inversion_NP2.append(frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png')
                        link_to_location_NP2.append(frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png')
                        link_to_semi_varigoram_2.append(frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png')
                        list_of_frames.append(frame)
                        list_of_dates.append(date) 


        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('Model Generated from Inversion NP2')):
            for ii, path in enumerate(link_to_inversion_NP2):
              if os.path.isfile(path):
                with doc.create(Figure(position='ht!')) as pic:
                                pic.add_image(path, width='350px')
                                pic.add_caption('Inversion results for NP2 for frame ' + str(list_of_frames[ii]) + ' on dates ' + list_of_dates[ii].split('/')[-1])

        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('Model Generated from USGS NP2')):
            for  ii, path in enumerate(link_to_forward_NP2):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Inversion results for NP2 for frame ' + str(list_of_frames[ii]) + ' on dates '  + list_of_dates[ii].split('/')[-1])


        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('Fault Plane Generated from Inversion NP2')):
            for  ii, path in enumerate(link_to_location_NP2):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Inversion results for NP2 for frame ' + str(list_of_frames[ii]) + ' on dates '  + list_of_dates[ii].split('/')[-1])


        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('Ifgms Used in Inversion')):
            for ii, path in enumerate(downsamp_mosaic_2):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Ifgms Used in Inversion from frame ' + str(active_frames[ii]) + ' on dates ' + list_of_dates[ii].split('/')[-1])

        doc.append(NoEscape(r'\FloatBarrier'))
        with doc.create(Subsection('SemiVariograms of Data used in Inversion')):
            for  ii, path in enumerate(link_to_semi_varigoram_2):
                if os.path.isfile(path):
                    with doc.create(Figure(position='ht!')) as pic:
                                    pic.add_image(path, width='350px')
                                    pic.add_caption('Inversion results for NP2 for frame ' + str(list_of_frames[ii]) + ' on dates '  + list_of_dates[ii].split('/')[-1])


        # for frame in active_frames:
        #     frame_dir_inv = os.path.join(GBIS_Area_path,'GEOC_' + frame + '_INVERSION_Results_NP2')
        #     frame_dir_proc = os.path.join(GBIS_Area_path,'GEOC_' + frame + '_processing_report_NP2')
        #     dates = [ f.path for f in os.scandir(frame_dir_inv) if f.is_dir() ]
        #     downsamp_mosaic = frame_dir_proc + '/All_ifgms_easy_look_up_downsamp.png'

        #     if os.path.isfile(downsamp_mosaic):
        #         with doc.create(Figure(position='h!')) as pic:
        #                     pic.add_image(downsamp_mosaic, width='350px')
        #                     pic.add_caption('Easy look up of all dates used in inversion.')
        #     doc.append(NoEscape(r'\FloatBarrier'))
        #     with doc.create(Subsection('USGS Forward Models for All Dates NP2')):
        #         for date in dates:
        #             # print(date)
        #             # print(active_ifgms)
        #             if date.split('/')[-1] in active_ifgms:
        #                 print('~~~~~~~~~~~~~~~~~~~~HERE~~~~~~~~~~~~~~~~~~~~~~~~')
        #                 link_to_forward_NP2 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_2.png'
        #                 link_to_inversion_NP2 =  frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png'
        #                 link_to_location_NP2 = frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png'
        #                 link_to_semi_varigoram = frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png'
        #                 if os.path.isfile(link_to_forward_NP2):
        #                     with doc.create(Figure(position='ht!')) as pic:
        #                         pic.add_image(link_to_forward_NP2, width='350px')
        #                         pic.add_caption('USGS forward model NP2 for dates ' + date.split('/')[-1] )
        #     doc.append(NoEscape(r'\FloatBarrier'))
        #     with doc.create(Subsection('Model Generated from Inversion NP2')):
        #         for date in dates:
        #             # print(date)
        #             # print(active_ifgms)
        #             if date.split('/')[-1] in active_ifgms:
        #                 print('~~~~~~~~~~~~~~~~~~~~HERE~~~~~~~~~~~~~~~~~~~~~~~~')
        #                 link_to_forward_NP2 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_2.png'
        #                 link_to_inversion_NP2 =  frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png'
        #                 link_to_location_NP2 = frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png'
        #                 link_to_semi_varigoram = frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png'
        #                 if os.path.isfile(link_to_inversion_NP2):
        #                     with doc.create(Figure(position='ht!')) as pic:
        #                         pic.add_image(link_to_inversion_NP2, width='350px')
        #                         pic.add_caption('Inversion results for NP2 for dates ' + date.split('/')[-1])
        #     doc.append(NoEscape(r'\FloatBarrier'))
        #     with doc.create(Subsection('Fault Plane Generated from Inversion NP2')):
        #         for date in dates:
        #             # print(date)
        #             # print(active_ifgms)
        #             if date.split('/')[-1] in active_ifgms:
        #                 print('~~~~~~~~~~~~~~~~~~~~HERE~~~~~~~~~~~~~~~~~~~~~~~~')
        #                 link_to_forward_NP2 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_2.png'
        #                 link_to_inversion_NP2 =  frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png'
        #                 link_to_location_NP2 = frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png'
        #                 link_to_semi_varigoram = frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png'
        #                 if os.path.isfile(link_to_location_NP2):
        #                     with doc.create(Figure(position='ht!')) as pic:
        #                         pic.add_image(link_to_location_NP2, width='350px')
        #                         pic.add_caption('Location results for NP2 for dates ' + date.split('/')[-1])
        #     doc.append(NoEscape(r'\FloatBarrier'))
        #     with doc.create(Subsection('Semivarigorams for NP2')):
        #         for date in dates:
        #             # print(date)
        #             # print(active_ifgms)
        #             if date.split('/')[-1] in active_ifgms:
        #                 print('~~~~~~~~~~~~~~~~~~~~HERE~~~~~~~~~~~~~~~~~~~~~~~~')
        #                 link_to_forward_NP2 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_2.png'
        #                 link_to_inversion_NP2 =  frame_dir_inv + '/' + date.split('/')[-1] + '/output_model_comp.png'
        #                 link_to_location_NP2 = frame_dir_inv + '/' + date.split('/')[-1] + '/2D_locations_plot.png'
        #                 link_to_semi_varigoram = frame_dir_proc +  '/' + 'semivarigram' + date.split('/')[-1] + 'X.png'
        #                 if os.path.isfile(link_to_semi_varigoram):
        #                     with doc.create(Figure(position='ht!')) as pic:
        #                         pic.add_image(link_to_semi_varigoram, width='350px')
        #                         pic.add_caption('Exponential fit Semivariogram for data with signal mask' + date.split('/')[-1] )

        

    doc.append(NoEscape(r'\FloatBarrier'))    
    with doc.create(Section('Frame info')):
        
        doc.append('Frames Passing Edge and Coherence Removals: ' + '\n')
        for frame in active_frames:
            # doc.append(frame + '\n')
            track_number = frame.split('_')[0].replace('D','').replace('A','') 
            if track_number.startswith('0'):
                track_number = track_number[1:len(track_number)]
            link_to_frame_jasmin = 'https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/' + track_number + '/' + frame
            doc.append(hyperlink(link_to_frame_jasmin,frame))
            doc.append('\n')
       
        for frame in active_frames:
            doc.append(NoEscape(r'\FloatBarrier'))
            with doc.create(Subsection(frame)):
               
                # doc.append(Subsection(frame))
                # doc.preamble.append(NoEscape(r'\clearpage'))
                doc.append('Dates of Coseismic ifgms: ' + '\n')
                frame_dir_inv = os.path.join(GBIS_Area_path,'GEOC_' + frame + '_INVERSION_Results_NP1')
                frame_dir_proc = os.path.join(GBIS_Area_path,'GEOC_' + frame + '_processing_report_NP1')


                track_number = frame.split('_')[0].replace('D','').replace('A','') 
                if track_number.startswith('0'):
                    track_number = track_number[1:len(track_number)]
                link_to_frame_jasmin = 'https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/' + track_number + '/' + frame

                # /uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/working_6/us7000fu12_GBIS_area/GEOC_166D_06139_131313_INVERSION_Results_NP1
                # print(frame_dir)
                # dates = next(os.walk(top_dir_NP1_inv))[1]
                dates = [ f.path for f in os.scandir(frame_dir_inv) if f.is_dir() ]
                if active_ifgms:
                    pass 
                else:
                    for date in dates:
                        active_ifgms.append(date.split('/')[-1]) 
                # print(dates)
                for date in dates:
                    link_to_date = link_to_frame_jasmin + '/interferograms/' + date.split('/')[-1]
                    link_to_forward_NP1 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_1.png'
                    link_to_forward_NP2 = frame_dir_proc +  '/' + date.split('/')[-1]+'forward_model_comp_2.png'
                    # print(frame_dir_proc)
                    # print(link_to_forward_NP1)
                    if active_ifgms: 
                        if date.split('/')[-1] in active_ifgms:
                            doc.append(TextColor('green',date.split('/')[-1]))
                            doc.append(hyperlink(link_to_date,'(Date used in inversion link)'))
                            doc.append('\n')
                        else:
                            doc.append(date.split('/')[-1])
                            doc.append(hyperlink(link_to_date,'(link)'))
                            doc.append('\n')
                    # else:
                    #     doc.append(TextColor('green',date.split('/')[-1]))
                    #     doc.append(hyperlink(link_to_date,'(Date used in inversion link)'))
                    #     doc.append('\n')
                

               
                    # with doc.create(Figure(position='h!')) as pic:
                    #     pic.add_image(link_to_forward_NP1, width='250px')
                    #     pic.add_caption('USGS forward model nodal plane one.')
                    #     # pic.add_image(link_to_forward_NP2, width='250px')
                    #     # pic.add_caption('USGS forward model nodal plane two.')

                link_to_easy_look_ifgms = frame_dir_proc + '/All_ifgms_easy_look_up_clipped.png'
                link_to_easy_look_mask = frame_dir_proc +  '/All_ifgms_easy_look_up_masked_signal_masked.png'
                link_to_easy_look_no_gac = frame_dir_proc +  '/All_ifgms_easy_look_up_coherence_masked.png'
                link_to_gacos_info = frame_dir_proc +  '/GACOS_info.png'
                link_to_network_png = frame_dir_proc + '/network11.png'

                if os.path.isfile(link_to_network_png):
                    with doc.create(Figure(position='ht!')) as pic:
                        pic.add_image(link_to_network_png, width='100px')
                        pic.add_caption('Network of available coseismic interferograms.')
                if os.path.isfile(link_to_easy_look_no_gac):
                    with doc.create(Figure(position='ht!')) as pic:
                        pic.add_image(link_to_easy_look_no_gac, width='350px')
                        pic.add_caption('All coseismic interferograms clipped around intial location run coherence mask applied but no GACOS correction.')
                if  os.path.isfile(link_to_gacos_info):
                    with doc.create(Figure(position='ht!')) as pic:
                        pic.add_image(link_to_gacos_info, width='350px')
                        pic.add_caption('GACOS Standard deviation reduction.')
                if os.path.isfile(link_to_easy_look_ifgms):
                    with doc.create(Figure(position='ht!')) as pic:
                        pic.add_image(link_to_easy_look_ifgms, width='350px')
                        pic.add_caption('All coseismic interferograms clipped around intial location run and with gacos correction and coherence mask applied.')
                if os.path.isfile(link_to_easy_look_mask):
                    with doc.create(Figure(position='ht!')) as pic:
                        pic.add_image(link_to_easy_look_mask, width='350px')
                        pic.add_caption('All coseismic interferograms with location run signal mask shown.')
                  
        



                
                    # pic.add_caption(')
            
    #     # with doc.create(Figure(position='h!')) as pic:
    #     #     pic.add_image(path_to_seismic_png, width='120px')
    #     #     pic.add_caption('USGS Seismically Derived Focal Mechanism')




    # print(inv_date_NP1)
            #
            #     doc.append('Report for processing of frame: ' + frame)
            
            
            #     for ii in range(len(pro_png_NP1)):
            #         
            #                     pic.add_image(pro_png_NP1[ii])
            #                     pic.add_caption(pro_png_NP1[ii].split('/')[-1].split('.')[0])

            #     with doc.create(Subsection('NP1_results')):
            #             for ii in range(len(NP1_results_png)):
            #                 with doc.create(Figure(position='h!')) as pic:
            #                         pic.add_image(NP1_results_png[ii])
            #                         if 'QAed' in NP2_results_png[ii].split('/')[-1].split('.')[0]:
            #                             pic.add_caption(NP2_results_png[ii].split('/')[-1].split('QAed')[-1].split('.')[0].replace('_',' '))
            #                         else:
            #                             pic.add_caption(NP2_results_png[ii].split('/')[-1].split('.')[0])
                        

            #             for ii in range(len(inv_date_NP1)):
            #                 for image in inv_date_NP1[ii]:
            #                     with doc.create(Figure(position='h!')) as pic:
            #                         pic.add_image(image)
            #                         pic.add_caption(image.split('/')[-1].split('.')[0])
                    
            #     with doc.create(Subsection('NP2_results')):
            #             for ii in range(len(NP2_results_png)):
            #                 with doc.create(Figure(position='h!')) as pic:
            #                     pic.add_image(NP2_results_png[ii])
            #                     if 'QAed' in NP2_results_png[ii].split('/')[-1].split('.')[0]:
            #                         pic.add_caption(NP2_results_png[ii].split('/')[-1].split('QAed')[-1].split('.')[0].replace('_',' '))
            #                     else:
            #                         pic.add_caption(NP2_results_png[ii].split('/')[-1].split('.')[0])
                        
            #             for ii in range(len(inv_date_NP2)):
            #                 for image in inv_date_NP2[ii]:
            #                     with doc.create(Figure(position='h!')) as pic:
            #                             pic.add_image(image)
            #                             pic.add_caption(image.split('/')[-1].split('.')[0])
                
            
   
        # print(pro_png_NP1)
        # print(pro_png_NP2)
        # print(inv_date_NP1)
        # print(inv_date_NP2)

    # doc.generate_pdf( clean_tex=True)
    
    doc.generate_pdf(GBIS_Area_path+'/Final_report_' + str(ID), clean_tex=False, compiler='pdflatex',silent=True)
    # doc.generate_pdf('test')
      

def hyperlink(url,text):
    text = escape_latex(text)
    return NoEscape(r'\href{' + url + '}{' + text + '}')



if __name__=='__main__':
    generateReport('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test/us6000b26j_GBIS_area','/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test/us6000b26j_NP1','/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test/us6000b26j_NP1','us6000b26j')
    
    # full_test = [
    #             'us60007anp',
    #             'us7000df40',
    #             'us6000bdq8',
    #             'us70006sj8', 
    #             'us7000g9zq',  
    #             'us70008cld',
    #             'us600068w0',
    #             'us7000m3ur',
    #             'us6000a8nh',
    #             'us7000mpe2',
    #             'us7000lt1i',
    #             'us6000b26j',
    #             'us6000ddge',
    #             'us7000mbuv',
    #             'us6000dxge',
    #             'us6000dyuk',
    #             'us6000jk0t',
    #             'us6000kynh',
    #             'us6000mjpj',
    #             'us7000abmk',
    #             'us7000cet5',
    #             'us7000fu12',
    #             'us7000gebb',
    # ]
    # full_test = ['us7000ljvg',
    #              'us6000iaqi',
    #              'us7000gx8m',      
    # ]
    # for ii in range(len(full_test)):
    #     USGS_ID = full_test[ii]
    #     try:
    #         GBIS_Area_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/' + USGS_ID + '_GBIS_area'
    #         GBIS_res_1 = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/'+ USGS_ID + '_NP1'
    #         GBIS_res_2 = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/'+ USGS_ID + '_NP2'
    #         generateReport(GBIS_Area_path,GBIS_res_1,GBIS_res_2,USGS_ID)
    #     except Exception as e:
    #         print(e)
        # generateReport(GBIS_Area_path,GBIS_res_1,GBIS_res_2,USGS_ID)

    