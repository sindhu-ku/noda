import argparse
import sys
import numpy as np

def fit_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pmt_opt',
                        type = str,
                        default = 'lpmt',
                        help = 'choose lpmt to use all PMTS and spmt for just the small ones')
    parser.add_argument('--stat_opt',
                        type=str,
                        default = '20years',
                        help = 'period used for sensitivity studies')
    parser.add_argument('--sin2_th13_opt',
                        type=str,
                        default = 'free',
                        help ='choose sin2_theta13 as fixed, free, or pull for the grid scan')
    parser.add_argument('--osc_formula_opt',
                        type = str,
                        default ="default",
                        help = 'choose oscillation formula, default or YB for yellowbook with m_ee')
    parser.add_argument('--PDG_opt',
                        type = str,
                        default ="PDG2019",
                        help = 'choose which values for parameters, Yellow Book (YB), PDG2016, PDG2018, PDG2019, Jinnan2019')
    parser.add_argument('--NMO_opt',
                        type = bool,
                        default =True,
                        help = 'choose which mass ordering, True for normal and vice versa')

    parser.add_argument('--me_rho',
                        type =float,
                        default = 2.45,
                        help= '(g/cm3) matter density, from the common inputs table')

    parser.add_argument('--min_ene',
                       type=float,
                       default = 0.8,
                       help='minimum energy for the spectra')
    parser.add_argument('--max_ene',
                       type=float,
                       default = 9.0,
                       help='maximum energy for the spectra')
    parser.add_argument('--bins',
                       type=int,
                       default = 411,
                       help='number of bins for the energy spectra')

    parser.add_argument('--ene_crop',
                        type=tuple,
                        default = (0.8, 9.0),
                        help='default energy range')
    parser.add_argument('--ene_crop2',
                        type=tuple,
                        default = (0.8, 12.0),
                        help='second energy range for calculating number of events')


    parser.add_argument('--FORCE_CALC_RM',
                        type=bool,
                        default = True,
                        help='option to calculate energy response matrix')
    parser.add_argument('--FORCE_CALC_CM',
                        type=bool,
                        default = True,
                        help='option to calculate covariance matrices')
    parser.add_argument('--PLOT_CM',
                        type=bool,
                        default = True,
                        help='option to plot covariance matrices')


    parser.add_argument('--input_data_file',
                        type=str,
                        default ="Data/JUNOInputs2022_05_08.root",
                        help='input root file with most inputs')
    parser.add_argument('--DYB_bump_file',
                        type=str,
                        default ="Data/JUNOInputs2022_05_08.root",
                        help='input root file with Daya Bay bump ratio')
    parser.add_argument('--reactor_model',
                        type=str,
                        default ="Huber-Mueller",
                        help='reactor model to be used')

    parser.add_argument('--cov_matrix_plots_folder',
                        type=str,
                        default ="CMplots_newbkg",
                        help='folder to save CM plots')
    parser.add_argument('--plots_folder',
                        type=str,
                        default ="CMplots_newbkg",
                        help='folder to save spectra plots')
    parser.add_argument('--data_matrix_folder',
                        type=str,
                        default ="Data_newbkg",
                        help='folder to save matrices')

    parser.add_argument('--grid',
                        type=dict,
                        default ={'sin2_12': np.linspace(0.302,0.312,21),
                        'sin2_13': np.linspace(0.0068,0.0368,21),
                        'dm2_21': np.linspace(7.47e-5,7.59e-5,21),
                        'dm2_31': np.linspace(2.515e-3,2.543e-3,21)},
                        help='grid space for the four parameters for the grid scan in form of a dictionary, order: sin2_th12, sin2_th13, dm2_21, dm2_31')

    parser.add_argument('--unc_list',
                        type=list,
                        default =['stat',
                        'eff',
                        'r2',
                        'crel',
                        'snf',
                        'noneq',
                        'b2b_DYB',
                        'abc',
                        'nl',
                        'bg',
                        'me',
                        'eff+r2+crel+snf+noneq+b2b_DYB+abc+nl+bg+me'],
                        help='list of combination of uncertainties to be considered for the grid scan')


    parser.add_argument('--sample_size_resp',
                        type=int,
                        default =10000,
                        help='sample size of response matrix fluctuations for CM')
    parser.add_argument('--sample_size_core',
                        type=int,
                        default =10000,
                        help='sample size of core powers fluctuations for CM')
    parser.add_argument('--sample_size_nonl',
                        type=int,
                        default =10000,
                        help='sample size of non-linearity fluctuations for CM')
    parser.add_argument('--sample_size_me',
                        type=int,
                        default =10000,
                        help='sample size of matter effect fluctuations for CM')
    return parser
