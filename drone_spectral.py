import numpy as np
import pylab as pl
import healpy as hp
import sys
from scipy import signal

arcmin2rad = 1.0 / 60 * np.pi / 180.0
Nside = 4096  # 1024#
I0 = 1e4
scan_type = 1  # 0. staring at the source 1. scanning at constant elevation
azel_source_input = (np.pi / 2, 45 * np.pi / 180.0)  # azel position of the source
t_v = np.arange(48, 52, 1 / 200)  # TOD time domain
delta_variations1 = 0.10  # 10% amplitude variations from the drone
delta_variations2 = 0.75  # 75% amplitude variations from the drone
variations_omega1 = 0.2  # small-time variations of the drone amplitude
variations_omega2 = 1.0  # strong-time variations of the drone amplitude
ipix_source_input = hp.ang2pix(Nside, azel_source_input[0], azel_source_input[1])
noise_level = I0 / 500
drone_amplitude = I0 / 50
beam_fwhm = 20  # arcmin
alpha_drone = 30. * np.pi / 180  # angle between the drone's grid and VPM
true_det_angle = 5. * np.pi / 180  # polarization angle of the detector

fhwp = 10.  # Hz, freq rotation of the HWP

# VPM parameters
z_0 = 6.e-3  # initial grid-mirror distance
f_vpm = 10.  # Hz, freq of the VPM


def chopper(t_, chopper_f=47., chopper_phi=0.0):
    """Return the chopper function for the input signal
    Args:
        t_ (float) : time in which the function is evaluated
        chopper_f (float) : chopper frequency
        chopper_phi (float) : chopper phase

    Returns :
    float : String value of the function for a time t_
    """
    return (signal.square(2 * np.pi * chopper_f * t_ + chopper_phi) + 1) / 2


def I_source(I0, drone_amplitude_loc, t_, chopper_f=47., chopper_phi=0.0, source_type='icvdn'):
    """ Return the sky signal depending on the input source type
    Args :
        I0 (float) : amplitude of the input signal
        drone_amplitude_loc (str) : amplitude of the thermal emission of the drone
        t_ (float) : time
        chopper_f (float) : chopper frequency
        chopper_phi (float) : chopper phase
        source_type (chr) : input source type

    Returns :
    float : value of the incident unpolarized light """

    if source_type == 'd':
        signal = 0.0
    else:
        signal = 1.0
    if 'i' in source_type:
        signal *= I0
    if 'c' in source_type:
        signal *= chopper(t_, chopper_f, chopper_phi)
    if 'v1' in source_type:
        signal *= (1. + np.cos(2 * np.pi * variations_omega1 * t_) * delta_variations1)
    if 'v2' in source_type:
        signal *= (1. + np.cos(2 * np.pi * variations_omega2 * t_) * delta_variations2)
    if 'd' in source_type:
        signal += drone_amplitude_loc
    return signal


def Mgrid():
    """Returns the Mueller matrix of a grid"""
    Mgrid_ = np.zeros((3, 3))
    Mgrid_[:2, :2] = 0.5
    return Mgrid_


def Mrot(alpha_):
    """Return the rotation matrix

    Args:
        alpha_ (float): angle of the rotation

    Returns :
        3*3 array : rotation matrix of angle 2*alpha
    """
    Mrot_ = np.eye(3)
    Mrot_[1, 1] = np.cos(2 * alpha_)
    Mrot_[1, 2] = np.sin(2 * alpha_)

    Mrot_[2, 1] = -np.sin(2 * alpha_)
    Mrot_[2, 2] = np.cos(2 * alpha_)
    return Mrot_


def pointing_matrix(t_, scan_type=1):
    """Return the direction of pointing of the telescope
    Args :
        t_ (float) :time
        scan_type (int) : scanning type of the sky

    Returns:
         pointing of the telescope in azimuth elevation

    """
    azel = np.zeros(2)
    if scan_type == 0:
        azel[0] = 0.0
        azel[1] = 45. * np.pi / 180
    elif scan_type == 1:
        azel[0] = np.pi / 2 + 10 * np.pi / 180 * np.cos(2 * np.pi * 0.02 / 4 * t_)
        # azel[0] = np.pi/2+ 0.1*np.cos(2*np.pi*0.1*t_)
        azel[1] = 45. * np.pi / 180
    else:
        print('i do not understand the scan type')
        sys.exit()
    return azel


def beam_conv(map_in, beam_fwhm):
    """Return the beam
    Args:
        map_in :
        beam_fwhm : beam width
    Returns :
        input signal map"""
    return hp.smoothing(map_in, beam_fwhm * arcmin2rad)


def Mhwp(delta=0):
    """Return the Mueller matrix of the hwp"""
    Mhwp_ = np.eye(3)
    Mhwp_[2, 2] = np.cos(delta)
    return Mhwp_


def Mvpm(delta=0):
    """Return the Mueller matrix of the VPM"""
    Mvpm_ = np.eye(3)
    Mvpm_[2, 2] = np.cos(delta)
    return Mvpm_


def z_computation(t_):
    """Return the grid-mirror distance function evoluting with time"""
    return z_0 * np.cos(2 * np.pi * f_vpm * t_)


def delta_computation(t_):
    """Return the delta VPM function evoluting with time"""
    return 4 * np.pi * 40e9 / 3e8 * z_computation(t_) * np.cos(0.0)


def source_mapping(t_, drone_amplitude_loc, m_in_conv,
                   azel_source=(0, 0), source_type='icvnd',
                   amplitude=I0):
    """Return the source signal
    Args:
        t_ (float) : time
        drone_amplitude_loc(float): amplitude of the thermal emission of the drone
        m_in_conv : input convoluted beam
        azel_source : azimuth elevation of the source
        source_type (chr) : input source type
         amplitude : amplitude of the input signal
    Returns:
        Stokes parameters of the polarized source signal I_source, Q_source, U_source
        """
    m_ = np.zeros((3, 12 * Nside ** 2))

    if azel_source != azel_source_input:
        ipix_source = hp.ang2pix(Nside, azel_source[0], azel_source[1])
    else:
        ipix_source = ipix_source_input

    source_type_ = ''
    if 'i' in source_type:
        source_type_ += 'i'
    if 'c' in source_type:
        source_type_ += 'c'
    if 'v1' in source_type:
        source_type_ += 'v1'
    if 'v2' in source_type:
        source_type_ += 'v2'

    m__ = m_in_conv * I_source(amplitude, drone_amplitude_loc, t_, source_type=source_type_)
    azel = pointing_matrix(t_, scan_type=scan_type)
    ipix_telescope = hp.ang2pix(Nside, azel[0], azel[1])

    observed_intensity = m__[:, ipix_telescope]
    M_rot_alpha_drone = Mrot(alpha_drone)
    M_polarizer_drone = Mgrid()

    Mdrone_ = M_rot_alpha_drone.T.dot(M_polarizer_drone).dot(M_rot_alpha_drone)

    observed_polarized_amplitude = Mdrone_.dot(observed_intensity)

    if 'd' in source_type:
        m__drone = m_in_conv * I_source(amplitude, drone_amplitude_loc, t_, source_type='d')
        observed_polarized_amplitude += m__drone[:, ipix_telescope]

    return observed_polarized_amplitude, ipix_telescope


def Mtot(alpha_det, t_, hwp='off', vpm='on'):
    """Return the total Mueller matrix of the instrument
    Args:
        alpha_det (float): angle of polarization of the detector
        t_ (float) : time
        hwp  : on if half-wave-plate considered as modulator
        vpm : on if VPM considered as modulator
    Returns :
        array : product of rotation and mueller matrices of each component of the instrument"""
    M_rot_alpha_det = Mrot(alpha_det)
    M_detector = Mgrid()
    M_ = np.eye(3)

    if (hwp == 'on') and (vpm == 'on'):
        print('you should choose hwp or vpm modulation')
        sys.exit()
    if (hwp == 'off') and (vpm == 'off'):
        M_ = Mrot(2 * np.pi * fhwp * 0.1).T.dot(Mhwp(np.pi)).dot(Mrot(2 * np.pi * fhwp * 0.1))
    if hwp == 'on':
        M_ = Mrot(2 * np.pi * fhwp * t_).T.dot(Mhwp(np.pi)).dot(Mrot(2 * np.pi * fhwp * t_))
    if vpm == 'on':
        M_ = Mvpm(delta=delta_computation(t_))

    Mfp_ = M_rot_alpha_det.T.dot(M_detector).dot(M_rot_alpha_det)

    return Mfp_.dot(M_), M_


def M_ana_builder(alpha_det, t_):
    """Return the analytical Mueller matrix of the instrument"""
    cosa = np.cos(2 * alpha_det)
    sina = np.sin(2 * alpha_det)
    cos4p = np.cos(4 * 2 * np.pi * fhwp * t_)
    sin4p = np.sin(4 * 2 * np.pi * fhwp * t_)

    M_ana_ = 0.5 * np.array([1.0, cosa * cos4p + sina * sin4p, cosa * sin4p - sina * cos4p])

    return M_ana_


def data_model(m_in_conv, true_det_angle_loc=true_det_angle,
               source_type='icvnd', amplitude=I0, t_v_loc=t_v, hwp='off', vpm='on', ana=False):
    """Return the model TOD
     Args :
         m_in_conv : convoluted input beam
         true_det_angle_loc (float) : angle of polarization of the detector
         source_type (chr) : input source type
         amplitude (float) : input amplitude of the signal
         t_v_loc : time domain of the TOD
         hwp : if hwp considered as modulator
         vpm : if vpm considered as modulator
         ana : if the analytical result returned

     Returns :
        data model with time dependency
     """
    data_ = []
    data_ana = []
    ind = 0

    for t_ in t_v_loc:
        s_in, idx = source_mapping(t_, m_in_conv, azel_source=azel_source_input,
                                   source_type=source_type, amplitude=I0)

        M, mhwp = Mtot(true_det_angle_loc, t_, hwp=hwp, vpm=vpm)
        Ms = M[0].dot(s_in)

        if ana:
            M_ana = M_ana_builder(true_det_angle_loc, t_)
            Ms_ana = M_ana.dot(s_in)
            data_.append(Ms)
            data_ana.append(Ms_ana)

        ind += 1

    if 'n' in source_type:
        np.random.seed(1)
        data_ = np.array(data_)
        data_ += np.random.normal(0, noise_level, size=data_.shape)

    return data_  # data_ana


# smooth input map for amplitude = 1
# print('beam convolution of the input map')
m_input_beam_conv = np.zeros((3, 12 * Nside ** 2))
m_input_beam_conv[0, ipix_source_input] = 1.0
m_input_beam_conv = beam_conv(m_input_beam_conv, beam_fwhm)
m_input_beam_conv /= np.max(m_input_beam_conv)
hp.write_map('m_input_beam_conv.fits', m_input_beam_conv, overwrite=True)
m_input_beam_conv = hp.read_map('m_input_beam_conv.fits', field=None)
np.save('m_input_beam.npy', m_input_beam_conv)
m_input_beam_conv = np.load('m_input_beam.npy')

# hp.gnomview(m_input_beam_conv[0], rot=(45,0.0,0.0), xsize=1000)


print('generating pointing, source intensity and data timestreams')
d_ = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='icdn')
d_i = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='i')
d_ic = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='ic')
d_icd = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='icd')
d_icv = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='icv1')
d_icn = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='icn')
d_icv1d = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='icv1d')
d_icv2d = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='icv2d')
d_n = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='n')
d_d = data_model(m_input_beam_conv, true_det_angle_loc=true_det_angle, source_type='d')

pl.figure()
pl.plot(t_v, d_, 'k-', label=r'$\alpha_{\rm det} = ' + str(round(true_det_angle * 180 / np.pi, 3)) + '\,\deg$')
pl.plot(t_v, d_i, 'DarkOrange', label='unchopped source')
pl.plot(t_v, d_ic, 'DarkBlue', label='chopped source')
pl.plot(t_v, d_icv, 'DarkBlue', linestyle='--', label='chopper-modulated signal w/ time variations')
pl.plot(t_v, d_n, 'DarkGray', linestyle='-', label='white noise')
pl.plot(t_v, d_d, 'DarkGreen', linestyle=':', label='drone thermal emission')
pl.ylabel('arbitrary units', fontsize=20)
pl.xlabel('time [sec]', fontsize=20)
pl.legend(frameon=True)
pl.savefig('TODall.pdf')
pl.show()

np.save('data_simulated_class.npy', d_)
np.save('data_simulated_class_ic', d_ic)
np.save('data_simulated_class_icd', d_icd)
np.save('data_simulated_class_icn', d_icn)
np.save('data_simulated_class_icv1d', d_icv1d)
np.save('data_simulated_class_icv2d', d_icv2d)

d_ = np.load('data_simulated_class.npy')
d_ic = np.load('data_simulated_class_ic.npy')
d_icn = np.load('data_simulated_class_icn.npy')
d_icd = np.load('data_simulated_class_icd.npy')
d_icv1d = np.load('data_simulated_class_icv1d.npy')
d_icv2d = np.load('data_simulated_class_icv2d.npy')

# template "drone parfait"

s_in_tot = []
for t_ in t_v:
    s_in, idx = source_mapping(t_, 1. / 50, m_input_beam_conv,
                               azel_source=azel_source_input,
                               source_type='icd', amplitude=1)
    s_in_tot.append(s_in)
np.save('s_in_tot.npy', s_in_tot)
s_in_tot = np.load('s_in_tot.npy')

# source Stokes parameters
# I = []
# Q = []
# U = [ ]
# for i in range (len(s_in_tot)):
#     I.append(s_in_tot[i][0])
#     Q.append(s_in_tot[i][1])
#     U.append(s_in_tot[i][2])

# pl.figure()
# pl.plot(t_v, I, label='I')
# pl.plot(t_v, Q, label='Q')
# pl.plot(t_v, U, label='U')
# pl.ylabel(r'$s_{in}$, arbitrary units', fontsize=20)
# pl.xlabel('time [sec]', fontsize=20)
# pl.legend(frameon=True)
# pl.savefig('stokes.pdf')
# pl.show()


t_v_loc = t_v

print('adjusting data arrays')
ind0 = list(t_v).index(t_v_loc[0])
ind1 = list(t_v).index(t_v_loc[-1])
d_loc = d_[ind0:ind1 + 1]
d_icn_loc = d_icn[ind0:ind1 + 1]
d_icd_loc = d_icd[ind0:ind1 + 1]
d_ic_loc = d_ic[ind0:ind1 + 1]
d_icv1d_loc = d_icv1d[ind0:ind1 + 1]
d_icv2d_loc = d_icv2d[ind0:ind1 + 1]


def chi2(alpha_det, s_source, plot=True):
    """ Return the chi2 function of the model et data
    Args :
        alpha_det : angle of polarization of the detector
        s_source : amplitude of the input signal
        plot : check the difference between the model and data
    Returns :
        chi2 function for different input source type"""

    ind = 0
    chi2_ = 0.0
    model = []
    data = []
    diff = []

    chi2_icn = 0.0
    chi2_ic = 0.0
    chi2_icd = 0.0
    chi2_icv1d = 0.0
    chi2_icv2d = 0.0

    for t_ in t_v:
        A_, _ = Mtot(alpha_det, t_, hwp='off', vpm='on')
        Ms = A_[0].dot(np.array(s_in_tot[ind])) * s_source

        model.append(Ms)
        data.append(d_[ind])
        diff.append(np.abs(model[ind] - d_[ind]))

        chi2_ += (d_[ind] - Ms) ** 2

        chi2_icn += (d_icn[ind] - Ms) ** 2
        chi2_ic += (d_ic[ind] - Ms) ** 2
        chi2_icd += (d_icd[ind] - Ms) ** 2
        chi2_icv1d += (d_icv1d[ind] - Ms) ** 2
        chi2_icv2d += (d_icv2d[ind] - Ms) ** 2

        ind += 1

    if noise_level != 0:
        chi2_ /= noise_level ** 2
        chi2_icn /= noise_level ** 2
        chi2_ic /= noise_level ** 2
        chi2_icd /= noise_level ** 2
        chi2_icv1d /= noise_level ** 2
        chi2_icv2d /= noise_level ** 2

    if plot:
        return model, diff

    return chi2_, chi2_icn, chi2_icd, chi2_ic, chi2_icv1d, chi2_icv2d


print('computing chi2 ... ')
a_v = np.linspace(0 * np.pi / 180, 10. * np.pi / 180, num=200)
s_v = np.linspace(9900, 10100, num=201)

for a in a_v:
    model, diff = chi2(a, I0, plot=True)

# 1D chi2 varying the angle of polarization of the detector, keeping a constant amplitude
# chi2_tot = np.zeros(len(a_v))
# chi2_icd_tot = np.zeros(len(a_v))
# chi2_ic_tot = np.zeros(len(a_v))
# chi2_icn_tot = np.zeros(len(a_v))
# chi2_icv1d_tot = np.zeros(len(a_v))
# chi2_icv2d_tot = np.zeros(len(a_v))
#
# ind_0 = 0
# for a in a_v:
#     chi2_tot[ind_0], chi2_icn_tot[ind_0], chi2_icd_tot[ind_0], chi2_ic_tot[ind_0], chi2_icv1d_tot[ind_0], \
#         chi2_icv2d_tot[ind_0] = chi2(a, I0, plot=False)
#     ind_0 += 1
#     print(ind_0 * 100. / len(a_v))

# 1D chi2 varying the input amplitude, keeping a constant alpha_det
# chi2_tot = np.zeros(len(s_v))
# chi2_ic_tot = np.zeros(len(s_v))
# chi2_icn_tot = np.zeros(len(s_v))
# chi2_icv1d_tot = np.zeros(len(s_v))
# chi2_icv2d_tot = np.zeros(len(s_v))
# ind_0 = 0
# for s in s_v:
#     chi2_tot[ind_0],chi2_icn_tot[ind_0],chi2_ic_tot[ind_0], chi2_icv1d_tot[ind_0], chi2_icv2d_tot[ind_0] = chi2(true_det_angle,s, plot=False)
#     ind_0 += 1
#     print(ind_0*100./len(s_v))


# pl.figure()
# pl.title('different sources TOD')
# # pl.plot(s_v, chi2_i_tot, label='source')
# pl.plot(a_v * 180 / np.pi, chi2_ic_tot, label='source + chopper')
# pl.plot(a_v * 180 / np.pi, chi2_icn_tot, label='source + choppern+ noise ')
# pl.plot(a_v * 180 / np.pi, chi2_icd_tot, label='source + choppern+ drone ')
# pl.plot(a_v * 180 / np.pi, chi2_tot, label='source + choppern+ noise+drone ')
# pl.plot(a_v * 180 / np.pi, chi2_icv1d_tot, label='source + chopper + drone + small time-var')
# pl.plot(a_v * 180 / np.pi, chi2_icv2d_tot, label='source + chopper + drone + strong time-var')
# pl.axvline(x=true_det_angle * 180 / np.pi, color='r', linestyle=':')
# pl.ylabel(r'$\chi^2$', fontsize=20)
# pl.xlabel(r'$\alpha_{det}$ [$arbitrary units$]', fontsize=20)
# pl.legend(loc='upper right', frameon=True, fontsize='20')
# pl.savefig('chivarioustod.pdf')
# pl.show()
#
# # likelihood
# pl.figure()
# # pl.plot(s_v, np.exp(-(np.array(chi2_i_tot)-np.min(np.array(chi2_i_tot)))/2),label='source')
# pl.plot(a_v * 180 / np.pi, np.exp(-(np.array(chi2_ic_tot) - np.min(np.array(chi2_ic_tot))) / 2), label='source+chopper')
# pl.plot(a_v * 180 / np.pi, np.exp(-(np.array(chi2_icn_tot) - np.min(np.array(chi2_icn_tot))) / 2),
#         label='source + chopper + noise')
# pl.plot(a_v * 180 / np.pi, np.exp(-(np.array(chi2_icd_tot) - np.min(np.array(chi2_icd_tot))) / 2),
#         label='source + chopper + drone')
# pl.plot(a_v * 180 / np.pi, np.exp(-(np.array(chi2_tot) - np.min(np.array(chi2_tot))) / 2),
#         label='source + chopper + noise + drone')
# pl.plot(a_v * 180 / np.pi, np.exp(-(np.array(chi2_icv1d_tot) - np.min(np.array(chi2_icv1d_tot))) / 2),
#         label='source + chopper + drone + small time-var')
# pl.plot(a_v * 180 / np.pi, np.exp(-(np.array(chi2_icv2d_tot) - np.min(np.array(chi2_icv2d_tot))) / 2),
#         label='source + chopper + drone + strong time-var')
# pl.ylabel(r'$likelihood$', fontsize=20)
# pl.xlabel(r'$\alpha_{det}$ [$aribtrary units$]', fontsize=20)
# pl.legend(loc='center left', frameon=True,fontsize=20 )
# pl.axvline(x=true_det_angle * 180 / np.pi, color='r', linestyle=':')
# pl.savefig('likelihoodvarioustod.pdf')
# pl.show()
# pl.figure()

ind_0 = 0
chi2_tot = np.zeros((len(a_v), len(s_v)))
chi2_i_tot = np.zeros((len(a_v), len(s_v)))
chi2_ic_tot = np.zeros((len(a_v), len(s_v)))
chi2_icn_tot = np.zeros((len(a_v), len(s_v)))
chi2_icv1d_tot = np.zeros((len(a_v), len(s_v)))
chi2_icv2d_tot = np.zeros((len(a_v), len(s_v)))
for a in a_v:
    ind_1 = 0
    for s in s_v:
        chi2_tot[ind_0, ind_1], chi2_icn_tot[ind_0, ind_1], chi2_ic_tot[ind_0, ind_1], chi2_icv1d_tot[ind_0, ind_1], \
            chi2_icv2d_tot[ind_0, ind_1] = chi2(a, s, plot=False)
        ind_1 += 1
    ind_0 += 1
    print(ind_0 * 100. / len(a_v))

np.save('chi2_tot', chi2_tot)
np.save('chi2_icn_tot', chi2_icn_tot)
np.save('chi2_ic_tot', chi2_ic_tot)
np.save('chi2_icv1d_tot', chi2_icv1d_tot)
np.save('chi2_icv2d_tot', chi2_icv2d_tot)

chi2_tot = np.load('chi2_tot.npy')
chi2_icn_tot = np.load('chi2_icn_tot.npy')
chi2_ic_tot = np.load('chi2_ic_tot.npy')
chi2_icv1d_tot = np.load('chi2_icv1d_tot.npy')
chi2_icv2d_tot = np.load('chi2_icv2d_tot.npy')

X, Y = np.meshgrid(a_v, s_v)
pl.title('total TOD')
pl.pcolor(X * 180 / np.pi, Y, chi2_tot.T)
pl.contour(X * 180 / np.pi, Y, chi2_tot.T, levels=[0., np.min(chi2_tot) + 2.30, np.min(chi2_tot) + 6.18], colors='k',
           linewidths=2.0)
pl.xlabel(r'$\alpha$ [$\deg$]', fontsize=20)
pl.axvline(x=true_det_angle * 180 / np.pi, color='r', linestyle=':')
pl.axhline(y=I0, color='r', linestyle=':')
pl.ylabel(r'source amplitude [arbitrary units]', fontsize=20)

pl.figure()
X, Y = np.meshgrid(a_v, s_v)
pl.title('various TOD sources')
CS1 = pl.contour(X * 180 / np.pi, Y, chi2_icn_tot.T,
                 levels=[0., np.min(chi2_icn_tot) + 2.30, np.min(chi2_icn_tot) + 6.18], colors='DarkGreen',
                 linewidths=2.0)
CS2 = pl.contour(X * 180 / np.pi, Y, chi2_ic_tot.T, levels=[0., np.min(chi2_ic_tot) + 2.30, np.min(chi2_ic_tot) + 6.18],
                 colors='DarkRed', linewidths=2.0)
CS3 = pl.contour(X * 180 / np.pi, Y, chi2_icv1d_tot.T,
                 levels=[0., np.min(chi2_icv1d_tot) + 2.30, np.min(chi2_icv1d_tot) + 6.18], colors='DarkOrange',
                 linewidths=2.0)
CS4 = pl.contour(X * 180 / np.pi, Y, chi2_icv2d_tot.T,
                 levels=[0., np.min(chi2_icv2d_tot) + 2.30, np.min(chi2_icv2d_tot) + 6.18], colors='DarkBlue',
                 linewidths=2.0)
CS5 = pl.contour(X * 180 / np.pi, Y, chi2_tot.T, levels=[0., np.min(chi2_tot) + 2.30, np.min(chi2_tot) + 6.18],
                 colors='DarkGreen', linewidths=2.0)
h1, _ = CS1.legend_elements()
h2, _ = CS2.legend_elements()
h3, _ = CS3.legend_elements()
h4, _ = CS4.legend_elements()
h5, _ = CS5.legend_elements()

labels = ['source + chopper+noise', 'source + chopper', 'source + chopper + drone + small time-variations',
          'source + chopper + drone + strong time-variations', 'source+chopper+drone+noise']
pl.legend([h1[0], h2[0], h3[0], h4[0], h5[0]], labels, frameon=True, fontsize='15', loc='upper left')

pl.xlabel(r'$\alpha$ [$\deg$]', fontsize=20)
pl.axvline(x=true_det_angle * 180 / np.pi, color='r', linestyle=':')
pl.axhline(y=I0, color='r', linestyle=':')
pl.ylabel(r'source amplitude [arbitrary units]', fontsize=20)
pl.savefig('chi2d_all.pdf')
pl.show()

# spectral likelihood
# def spectral_likelihood(alpha_det):
#     # logL = 0.0

#     # t_v_loc = t_v[880:920:1]
#     t_v_loc = t_v[:]

#     ind = 0
#     # A = np.zeros((len(t_v_loc),3))
#     A = np.zeros((len(t_v_loc),2))
#     # Aana = np.zeros((len(t_v_loc),3))
#     # AtNinvA_ = np.zeros((3,3))
#     AtNinvA_ = np.zeros((2,2))
#     d_loc = []how
#     for t_ in t_v_loc :
#         A_, _ = Mtot(alpha_det, t_)
#         A[ind,:] = A_[0][1:]

#         # A_ana = M_ana_builder(alpha_det, t_)
#         # Aana[ind,:] = A_ana

#         d_loc.append(d_icn[list(t_v_loc).index(t_)])

#         ind += 1

#     # pl.figure()
#     # pl.plot(t_v_loc, d_loc)
#     # pl.show()
#     # sys.exit()

#     AtNinvA_ = A.T.dot(A)
#     # AtNinvA_ana = Aana.T.dot(Aana)

#     # breakpoint()

#     if noise_level != 0:
#         AtNinvd = A.T.dot(np.array(d_loc))/noise_level**2
#         AtNinvA = AtNinvA_/noise_level**2
#     else:
#         AtNinvd = A.T.dot(np.array(d_loc))
#         AtNinvA = AtNinvA_

#     # AtNinvA = AtNinvA.T
#     # breakpoint()

#     try:
#         invAtNinvA = np.linalg.inv(np.diag(np.diag(AtNinvA))) # forcing it to be diagonal ...
#         # invAtNinvA = np.linalg.inv(AtNinvA)
#     except:
#         invAtNinvA = np.zeros_like(AtNinvA)
#     # print(invAtNinvA)
#     # breakpoint()
#     # invAtNinvA = np.diag(np.diag(invAtNinvA))
#     logL = AtNinvd.T.dot(invAtNinvA).dot(AtNinvd)

#     # logL1 = (A[:,1].dot(np.array(d_loc)))**2/AtNinvA_[1,1]
#     logL1 = (A[:,0].dot(np.array(d_loc)))**2/AtNinvA_[0,0]
#     # logL2 = (A[:,2].dot(np.array(d_loc)))**2/AtNinvA_[2,2]
#     logL2 = (A[:,1].dot(np.array(d_loc)))**2/AtNinvA_[1,1]
#     # logL3 = (A[:,2].dot(d_))**2/AtNinvA_[1,1]
#     # logL4 = (A[:,1].dot(d_))**2/AtNinvA_[2,2]
#     # logL = AtNinvd.T.dot(AtNinvd)
#     '''
#     ind = 0
#     # A = np.zeros((len(t_v),3))
#     # AtNinvA_ = np.zeros((3,3))
#     logL = 0
#     for t_ in t_v :
#         A_, _ = Mtot(alpha_det, t_)
#         # A[ind,:] = A_[0]
#         if noise_level != 0:
#             AtNinvA_ = A_[0].T.dot(A_[0])/noise_level**2
#             AtNinvd = A_[0].T.dot(d_[ind])/noise_level**2
#         else:
#             AtNinvA_ = A_[0].T.dot(A_[0])
#             AtNinvd = A_[0].T.dot(d_[ind])

#         logL += AtNinvd.T.dot(AtNinvd)/AtNinvA_

#         ind += 1

#     print(alpha_det, logL)
#     '''

#     return logL, A, logL1, logL2#, AtNinvd, invAtNinvA, logL3, logL4

# a_v = np.linspace(-20.*np.pi/180, 20.*np.pi/180, num=500)
# a_v = [-10*np.pi/180,0,15*np.pi/180]
# logL_, A = [spectral_likelihood(a) for a in a_v]
# logL_ = []
# A_ = []
# term1_ = []
# term2_ = []
# logL1_ = []
# logL2_ = []
# logL3_ = []
# logL4_ = []
# for a in a_v:
#     # logL__, A__, x1, x2, l1, l2, l3, l4 = spectral_likelihood(a)
#     logL__, _ , l1, l2 = spectral_likelihood(a)
#     logL_.append(logL__)
#     # A_.append(A__)
#     # term1_.append(x1)
#     # term2_.append(x2)
#     logL1_.append(l1)
#     logL2_.append(l2)
#     # logL3_.append(l3)
#     # logL4_.append(l4)

# pl.plot(a_v*180/np.pi, logL_)
# pl.plot(a_v*180/np.pi, logL1_, 'k--')
# pl.plot(a_v*180/np.pi, np.exp( -(np.array(logL2_)-np.min(logL2_))), 'k-')
# pl.plot(a_v*180/np.pi, np.exp( -(np.array(logL_)-np.min(logL_))), 'k-')
# pl.plot(a_v, logL3_, 'k:')
# pl.plot(a_v, logL4_, 'r:')
# pl.axvline(x=true_det_angle*180/np.pi, color='r', linestyle='--')
# pl.show()


# t10 = []
# t11 = []
# t12 = []
# tinv = []
# for t in term1_:
#     t10.append(t[0])
#     t11.append(t[1])
#     t12.append(t[2])
# for t in term2_:
#     tinv.append(np.linalg.det(t))


sys.exit()
