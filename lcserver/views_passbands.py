"""Photometric conversions, written out and calculable.

Another standalone utility, tied to no processing at all: the combined light
curve is drawn on a single scale, and a dozen surveys have to be brought onto
it. This is the arithmetic that does it, together with the relations stdpipe
uses when it augments a catalogue, each with where it comes from.

Every relation here has the same shape - the magnitude that goes in, plus a
polynomial in one or more colours:

    out = in + sum_k poly_k(colour_k)

which is what lets one calculator serve all of them, in the page as well as
here. The coefficients are in numpy.polyval order, highest power first.
"""

from django.template.response import TemplateResponse

from collections import OrderedDict


# Where a relation is used, and so how much it is worth trusting
GROUPS = OrderedDict([
    ('lcserver', {
        'name': 'Light curve conversions',
        'note': "What the combined light curve uses to put every survey on the "
                "Pan-STARRS g scale. All but one assume a single fixed colour for "
                "the star, taken from the info step - a model rather than a "
                "measurement, and one that costs most exactly where a light curve "
                "is most interesting, when a variable changes colour as it changes "
                "brightness.",
    }),
    ('stdpipe', {
        'name': 'Catalogue conversions (stdpipe)',
        'note': "What stdpipe.catalogs.augment_cat_bands() adds to a catalogue "
                "when it is fetched, so that a frame may be calibrated against "
                "bands the catalogue never published. These act on catalogue "
                "stars, whose colours are measured rather than assumed.",
    }),
])


def conversion(id, title, group, base, output, terms=(), offset=0.0,
               reference=None, url=None, used_by=None, valid=None, sigma=None,
               note=None):
    """One relation, as the page shows it and as the calculator evaluates it.

    `terms` pairs a colour with the polynomial coefficients applied to it, and
    `offset` is what is added regardless of any colour.
    """
    return {
        'id': id,
        'title': title,
        'group': group,
        'base': base,
        'output': output,
        'terms': [{'color': color, 'coeffs': list(coeffs)} for color, coeffs in terms],
        'offset': offset,
        'reference': reference,
        'url': url,
        'used_by': used_by,
        'valid': [{'color': color, 'min': lo, 'max': hi} for color, lo, hi in (valid or [])],
        'sigma': sigma,
        'note': note,
    }


# Papers behind more than one relation, so that the same one is cited the same
# way everywhere
REF_TONRY = ('Tonry et al. (2012), ApJ 750, 99, Table 6', 'https://arxiv.org/abs/1203.0297')
REF_KOSTOV = ('Kostov & Bonev (2017), Bulg. Astron. J. 28, 3',
              'https://arxiv.org/abs/1706.06147')
REF_LUPTON = ('Lupton (2005), as published by SDSS',
              'https://www.sdss.org/dr12/algorithms/sdssUBVRITransform/#Lupton2005')
REF_PANCINO = ('fitted on the curated Landolt and Stetson collections of '
               'Pancino et al. (2022), A&A 664, A109', 'https://arxiv.org/abs/2205.06186')
REF_GAIA_DR2 = ('Gaia DR2 documentation, photometric relationships',
                'https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/'
                'chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html')
REF_GAIA_EDR3 = ('Gaia EDR3 documentation, Table 5.6',
                 'https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/'
                 'chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html')


CONVERSIONS = [
    # -- What the light curves are built from ------------------------------
    conversion(
        'v_to_g', 'Johnson V to Pan-STARRS g', 'lcserver', 'V', 'g',
        terms=[('g - r', [0.008, 0.498, 0.02])],
        reference=REF_KOSTOV[0] + ', inverted', url=REF_KOSTOV[1],
        used_by='ASAS-SN V, ASAS-3, CSS, KWS, Hipparcos, INTEGRAL OMC, NSVS',
        note="Published the other way round, as V = g - 0.02 - 0.498*(g - r) "
             "- 0.008*(g - r)^2, and inverted here.",
    ),
    conversion(
        'b_to_g', 'Johnson B to Pan-STARRS g', 'lcserver', 'B', 'g',
        terms=[('g - r', [-0.3130, -0.2271])],
        reference=REF_LUPTON[0] + ', inverted', url=REF_LUPTON[1],
        used_by='KWS B', sigma=0.011,
        note="The scatter quoted is the one of the original fit. The photographic "
             "plates this is used on are a good deal further from Johnson B than "
             "that, so the number to expect is the plates' own colour term.",
    ),
    conversion(
        'r_to_g', 'Pan-STARRS r to Pan-STARRS g', 'lcserver', 'r', 'g',
        terms=[('g - r', [1.0, 0.0])],
        reference='the definition of the colour',
        used_by='Pan-STARRS r epochs, in the info step',
    ),
    conversion(
        'gaia_g_to_g', 'Gaia G to Pan-STARRS g', 'lcserver', 'G', 'g',
        terms=[('BP - RP', [-0.0064, 0.1548, 0.6365, -0.2199])],
        reference=REF_GAIA_EDR3[0], url=REF_GAIA_EDR3[1],
        used_by='Gaia epoch photometry, in the info step',
        valid=[('BP - RP', 0.3, 3.0)], sigma=0.0745,
        note="G is converted rather than BP, which sits closer to g and would need "
             "a smaller correction: G is the more precisely measured of the two, and "
             "its relation is the better determined.",
    ),
    conversion(
        'rotse_to_v', 'Unfiltered ROTSE-I to Johnson V', 'lcserver', 'm_ROTSE', 'V',
        terms=[('B - V', [1/1.875, 0.0])],
        reference='Wozniak et al. (2004), AJ 127, 2436',
        url='https://arxiv.org/abs/astro-ph/0401217',
        used_by='NSVS, which is then taken through V to g',
        note="The NSVS magnitudes are defined against V with a colour term already "
             "in them, m_ROTSE = V - (B - V)/1.875, so the band is on the V scale "
             "for a star of zero colour and drifts from it for any other.",
    ),
    conversion(
        'asas_g_to_ps1_g', 'ASAS-SN g to Pan-STARRS g', 'lcserver', 'g (SDSS)', 'g (PS1)',
        terms=[('g - r', [-0.019, -0.145, -0.013])],
        reference=REF_TONRY[0] + ', inverted', url=REF_TONRY[1],
        used_by='ASAS-SN g',
        note="ASAS-SN g is on the SDSS scale, and this is the PS1-to-SDSS relation "
             "below, run backwards.",
    ),
    conversion(
        'applause_bp_to_g', 'Gaia BP to Pan-STARRS g', 'lcserver', 'BP', 'g',
        terms=[('g - r', [0.11445168305534677, 0.20378930951540578, -0.0499368274565225])],
        reference='fitted on Landolt standards',
        used_by='APPLAUSE photographic plates',
        note="The plates arrive as natural magnitudes with a per-plate colour term, "
             "which is removed as RP = natmag - (BP - RP)*colour_term; BP then "
             "follows from an assumed constant (BP - RP), and this puts it onto g.",
    ),
    conversion(
        'applause_bp_to_r', 'Gaia BP to Pan-STARRS r', 'lcserver', 'BP', 'r',
        terms=[('g - r', [0.13189831407771777, -0.8213890428750275, -0.04388161680503415])],
        reference='fitted on Landolt standards',
        used_by='APPLAUSE photographic plates',
    ),

    # -- What stdpipe adds to a catalogue ----------------------------------
    conversion(
        'ps1_to_B', 'Pan-STARRS to Johnson B', 'stdpipe', 'g', 'B',
        terms=[('g - r', [0.10339527794499666, -0.492149523946056, 1.2093816061394638,
                          0.061925048331498395]),
               ('r - i', [-0.2571974580267897, 0.9211495207523038, -0.8243222108864755,
                          0.0619250483314976])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1],
        used_by='PS1 and ATLAS-RefCat2 catalogues',
        valid=[('g - r', -0.5, 2.5), ('r - i', -0.5, 2.0)],
    ),
    conversion(
        'ps1_to_V', 'Pan-STARRS to Johnson V', 'stdpipe', 'g', 'V',
        terms=[('g - r', [-0.011452922062676726, -9.949308251868327e-05,
                          -0.4650511584366353, -0.007076854914511554]),
               ('r - i', [0.012749150754020416, 0.057554580469724864,
                          -0.09019328095355343, -0.007076854914511329])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1],
        used_by='PS1 and ATLAS-RefCat2 catalogues',
        valid=[('g - r', -0.5, 2.5), ('r - i', -0.5, 2.0)],
    ),
    conversion(
        'ps1_to_R', 'Pan-STARRS to Cousins R', 'stdpipe', 'r', 'R',
        terms=[('g - r', [0.004905242602502597, -0.046545625824660514,
                          0.07830702317352654, -0.08438139204305026]),
               ('r - i', [-0.07782426914647306, 0.14090289318728444,
                          -0.3634922073369279, -0.08438139204305031])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1],
        used_by='PS1 and ATLAS-RefCat2 catalogues',
        valid=[('g - r', -0.5, 2.5), ('r - i', -0.5, 2.0)],
    ),
    conversion(
        'ps1_to_I', 'Pan-STARRS to Cousins I', 'stdpipe', 'i', 'I',
        terms=[('g - r', [-0.02239162647929074, 0.04401240100377888,
                          -0.038500349283596795, -0.19509051168348646]),
               ('r - i', [0.014586929059030904, -0.025228407778416825,
                          -0.21476143248697746, -0.19509051168348637])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1],
        used_by='PS1 and ATLAS-RefCat2 catalogues',
        valid=[('g - r', -0.5, 2.5), ('r - i', -0.5, 2.0)],
    ),
    conversion(
        'ps1_to_sdss_g', 'Pan-STARRS g to SDSS g', 'stdpipe', 'g (PS1)', 'g (SDSS)',
        terms=[('g - r', [0.019, 0.145, 0.013])],
        reference=REF_TONRY[0], url=REF_TONRY[1],
        used_by='PS1 and ATLAS-RefCat2 catalogues',
    ),
    conversion(
        'ps1_to_sdss_r', 'Pan-STARRS r to SDSS r', 'stdpipe', 'r (PS1)', 'r (SDSS)',
        terms=[('g - r', [0.007, 0.004, -0.001])],
        reference=REF_TONRY[0], url=REF_TONRY[1],
        used_by='PS1 and ATLAS-RefCat2 catalogues',
    ),
    conversion(
        'ps1_to_sdss_i', 'Pan-STARRS i to SDSS i', 'stdpipe', 'i (PS1)', 'i (SDSS)',
        terms=[('g - r', [0.010, 0.011, -0.005])],
        reference=REF_TONRY[0], url=REF_TONRY[1],
        used_by='PS1 and ATLAS-RefCat2 catalogues',
        note="z is taken over unchanged.",
    ),

    conversion(
        'skymapper_to_ps1_g', 'SkyMapper DR4 to Pan-STARRS g', 'stdpipe', 'g (SkyMapper)', 'g (PS1)',
        terms=[('g - r', [-0.07715320986152466, 0.2694597282089696, 0.04069379065128178,
                          0.01396290714542747]),
               ('r - i', [0.026097008342026252, -0.14040957287568073, 0.133647539780504,
                          0.013962907145427432])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
    ),
    conversion(
        'skymapper_to_ps1_r', 'SkyMapper DR4 to Pan-STARRS r', 'stdpipe', 'r (SkyMapper)', 'r (PS1)',
        terms=[('g - r', [0.08779280979185472, -0.23257704629617004, 0.1890698144343673,
                          -0.008125550119663026]),
               ('r - i', [-0.06273832689338121, 0.21909317812693613, -0.23340488268623696,
                          -0.00812555011966309])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
    ),
    conversion(
        'skymapper_to_ps1_i', 'SkyMapper DR4 to Pan-STARRS i', 'stdpipe', 'i (SkyMapper)', 'i (PS1)',
        terms=[('g - r', [0.03553380678975111, -0.021174189684500792,
                          -0.028159666883815007, 0.0009748746568893062]),
               ('r - i', [0.00911922467970264, -0.0362286983251751, 0.1403094994141109,
                          0.0009748746568892609])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
    ),
    conversion(
        'skymapper_to_ps1_z', 'SkyMapper DR4 to Pan-STARRS z', 'stdpipe', 'z (SkyMapper)', 'z (PS1)',
        terms=[('g - r', [0.08071260245520126, -0.051693023216670575, -0.0739439627982131,
                          -0.0020460270205769223]),
               ('r - i', [0.09720715174271254, -0.32063637962189184, 0.37918283208242526,
                          -0.0020460270205769305])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
    ),
    conversion(
        'skymapper_to_ps1_y', 'SkyMapper DR4 to Pan-STARRS y', 'stdpipe', 'z (SkyMapper)', 'y (PS1)',
        terms=[('g - r', [0.038781034592287725, -0.11040188064275973, 0.08235396198116865,
                          0.006980454415779221]),
               ('r - i', [-0.0649739656901001, 0.205320995228645, -0.28233276303592,
                          0.006980454415779424])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
        note="SkyMapper has no y band of its own, so y is built from its z.",
    ),
    conversion(
        'skymapper_to_B', 'SkyMapper DR4 to Johnson B', 'stdpipe', 'g (SkyMapper)', 'B',
        terms=[('g - r', [-0.22773918482205113, 0.1818124624962873, 1.0021365492384895,
                          0.10762635377473588]),
               ('r - i', [-0.004034933919297649, 0.08214592357213418,
                          -0.07535454054888649, 0.10762635377473558])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
    ),
    conversion(
        'skymapper_to_V', 'SkyMapper DR4 to Johnson V', 'stdpipe', 'g (SkyMapper)', 'V',
        terms=[('g - r', [-0.02545732895304914, 0.03256423830249228,
                          -0.33074199873567045, -0.002938730214382037]),
               ('r - i', [-0.007342074336918033, 0.08255055271047995,
                          -0.14349325478829064, -0.0029387302143822])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
    ),
    conversion(
        'skymapper_to_R', 'SkyMapper DR4 to Cousins R', 'stdpipe', 'r (SkyMapper)', 'R',
        terms=[('g - r', [0.07296699439306827, -0.1943702618426095, 0.15375263988851387,
                          -0.08547735652048871]),
               ('r - i', [-0.07378125129406726, 0.18462924970775316,
                          -0.40720945890364135, -0.08547735652048903])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
    ),
    conversion(
        'skymapper_to_I', 'SkyMapper DR4 to Cousins I', 'stdpipe', 'i (SkyMapper)', 'I',
        terms=[('g - r', [-0.00925391710305653, 0.046223960182760516,
                          -0.06889215990613289, -0.19321699685334734]),
               ('r - i', [0.01197866152020802, -0.044370062623186206,
                          -0.05231484699406009, -0.19321699685334745])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1], used_by='SkyMapper DR4 catalogue',
    ),

    conversion(
        'gaiadr2_to_B', 'Gaia DR2 to Johnson B', 'stdpipe', 'G', 'B',
        terms=[('BP - RP', [-0.05927724559795761, 0.4224326324292696, 0.626219707920836,
                            -0.011211539139725953]),
               ('C*', [876.4047401692277, 5.114021693079334, -2.7332873314449326, 0])],
        reference='fitted on Stetson standards', used_by='Gaia DR2 catalogue',
        note="C* is the corrected BP/RP flux excess, which measures how much a source "
             "departs from a clean point source; it is zero for well behaved stars, "
             "and the calculator leaves it there unless you say otherwise. G itself is "
             "corrected for the DR2 saturation effects first.",
    ),
    conversion(
        'gaiadr2_to_V', 'Gaia DR2 to Johnson V', 'stdpipe', 'G', 'V',
        terms=[('BP - RP', [0.0017624722901609662, 0.15671377090187089,
                            0.03123927839356175, 0.041448557506784556]),
               ('C*', [98.03049528983964, 20.582521666713028, 0.8690079603974803, 0])],
        reference='fitted on Stetson standards', used_by='Gaia DR2 catalogue',
    ),
    conversion(
        'gaiadr2_to_R', 'Gaia DR2 to Cousins R', 'stdpipe', 'G', 'R',
        terms=[('BP - RP', [0.02045449129406191, 0.054005149296716175,
                            -0.3135475489352255, 0.020545083667168156]),
               ('C*', [347.42190542330945, 39.42482430363565, 0.8626828845232541, 0])],
        reference='fitted on Stetson standards', used_by='Gaia DR2 catalogue',
    ),
    conversion(
        'gaiadr2_to_I', 'Gaia DR2 to Cousins I', 'stdpipe', 'G', 'I',
        terms=[('BP - RP', [0.005092289380850884, 0.07027022935721515,
                            -0.7025553064161775, -0.02747532184796779]),
               ('C*', [79.4028706486939, 9.176899238787003, -0.7826315256072135, 0])],
        reference='fitted on Stetson standards', used_by='Gaia DR2 catalogue',
    ),
    conversion(
        'gaiadr2_to_sdss_g', 'Gaia DR2 G to SDSS g', 'stdpipe', 'G', 'g (SDSS)',
        terms=[('BP - RP', [-0.021349, 0.25171, 0.46245, -0.13518])],
        reference=REF_GAIA_DR2[0], url=REF_GAIA_DR2[1], used_by='Gaia DR2 catalogue',
    ),
    conversion(
        'gaiadr2_to_sdss_r', 'Gaia DR2 G to SDSS r', 'stdpipe', 'G', 'r (SDSS)',
        terms=[('BP - RP', [0.049465, 0.027464, -0.24662, 0.12879])],
        reference=REF_GAIA_DR2[0], url=REF_GAIA_DR2[1], used_by='Gaia DR2 catalogue',
    ),
    conversion(
        'gaiadr2_to_sdss_i', 'Gaia DR2 G to SDSS i', 'stdpipe', 'G', 'i (SDSS)',
        terms=[('BP - RP', [0.10141, -0.64728, 0.29676])],
        reference=REF_GAIA_DR2[0], url=REF_GAIA_DR2[1], used_by='Gaia DR2 catalogue',
    ),
    conversion(
        'gaiadr2_to_ps1_g', 'Johnson B to Pan-STARRS g', 'stdpipe', 'B', 'g (PS1)',
        terms=[('B - V', [-0.032, -0.485, -0.108])],
        reference='fitted here, on the Johnson magnitudes converted above',
        used_by='Gaia DR2 catalogue',
        note="Flagged in stdpipe as still carrying uncorrected colour and magnitude "
             "trends.",
    ),
    conversion(
        'gaiadr2_to_ps1_r', 'Johnson V to Pan-STARRS r', 'stdpipe', 'V', 'r (PS1)',
        terms=[('B - V', [0.041, -0.462, 0.082])],
        reference='fitted here, on the Johnson magnitudes converted above',
        used_by='Gaia DR2 catalogue',
        note="Flagged in stdpipe as still carrying uncorrected colour and magnitude "
             "trends.",
    ),

    conversion(
        'sdss_to_ps1_g', 'SDSS g to Pan-STARRS g', 'stdpipe', 'g (SDSS)', 'g (PS1)',
        terms=[('g - r', [-0.030414391501015867, -0.09960002492299584,
                          -0.002910024005294562])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1],
        used_by='Gaia DR3 synthetic photometry, which is published on the SDSS scale',
    ),
    conversion(
        'sdss_to_ps1_r', 'SDSS r to Pan-STARRS r', 'stdpipe', 'r (SDSS)', 'r (PS1)',
        terms=[('g - r', [-0.009566553708653305, 0.014924591443344211,
                          -0.003928147919030857])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1],
        used_by='Gaia DR3 synthetic photometry',
    ),
    conversion(
        'sdss_to_ps1_i', 'SDSS i to Pan-STARRS i', 'stdpipe', 'i (SDSS)', 'i (PS1)',
        terms=[('g - r', [-0.010802807724098494, 0.01124900218746879,
                          0.01274293783734852])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1],
        used_by='Gaia DR3 synthetic photometry',
    ),
    conversion(
        'sdss_to_ps1_z', 'SDSS z to Pan-STARRS z', 'stdpipe', 'z (SDSS)', 'z (PS1)',
        terms=[('g - r', [-0.0031896767661109523, 0.06537983414287968,
                          0.007695587806229381])],
        reference=REF_PANCINO[0], url=REF_PANCINO[1],
        used_by='Gaia DR3 synthetic photometry',
    ),

    conversion(
        'apass_to_R', 'APASS r to Cousins R', 'stdpipe', 'r (APASS)', 'R',
        terms=[('g - r', [-0.014, -0.087, -0.157])],
        reference='fitted on Landolt standards', used_by='APASS DR9 catalogue',
    ),
    conversion(
        'apass_to_I', 'APASS i to Cousins I', 'stdpipe', 'i (APASS)', 'I',
        terms=[('g - r', [-0.004, -0.118, -0.354])],
        reference='fitted on Landolt standards', used_by='APASS DR9 catalogue',
    ),

    conversion(
        'vista_to_2mass_J', 'VISTA J to 2MASS J', 'stdpipe', 'J (VISTA)', 'J (2MASS)',
        terms=[('J - H', [0.070, 0.0])],
        reference='A&A 552, A101 (2013)',
        url='https://ui.adsabs.harvard.edu/abs/2013A%26A...552A.101S',
        used_by='VHS catalogue',
        note="A star without the second band keeps its uncorrected VISTA magnitude "
             "rather than losing it.",
    ),
    conversion(
        'vista_to_2mass_H', 'VISTA H to 2MASS H', 'stdpipe', 'H (VISTA)', 'H (2MASS)',
        terms=[('H - Ks', [-0.035, 0.0])],
        reference='A&A 552, A101 (2013)',
        url='https://ui.adsabs.harvard.edu/abs/2013A%26A...552A.101S',
        used_by='VHS catalogue',
    ),
    conversion(
        'vista_to_2mass_Ks', 'VISTA Ks to 2MASS Ks', 'stdpipe', 'Ks (VISTA)', 'Ks (2MASS)',
        terms=[('J - Ks', [-0.011, 0.0])],
        reference='A&A 552, A101 (2013)',
        url='https://ui.adsabs.harvard.edu/abs/2013A%26A...552A.101S',
        used_by='VHS catalogue',
    ),

    conversion(
        'sdss_u_to_ab', 'SDSS u to AB', 'stdpipe', 'u (SDSS)', 'u (AB)', offset=-0.04,
        reference='SDSS DR16 flux calibration notes',
        url='https://www.sdss4.org/dr16/algorithms/fluxcal/#SDSStoAB',
        used_by='SDSS DR16 catalogue',
        note="The SDSS system is not quite AB at the ends of its wavelength range.",
    ),
    conversion(
        'sdss_z_to_ab', 'SDSS z to AB', 'stdpipe', 'z (SDSS)', 'z (AB)', offset=0.02,
        reference='SDSS DR16 flux calibration notes',
        url='https://www.sdss4.org/dr16/algorithms/fluxcal/#SDSStoAB',
        used_by='SDSS DR16 catalogue',
    ),
]


def format_term(color, coeffs):
    """One polynomial in one colour, as it is written on the page."""
    parts = []
    power = len(coeffs) - 1

    for coeff in coeffs:
        if coeff:
            value = '%.5g' % abs(coeff)

            if power == 0:
                term = value
            elif power == 1:
                term = ('' if value == '1' else value + '*') + '(%s)' % color
            else:
                term = ('' if value == '1' else value + '*') + '(%s)^%d' % (color, power)

            parts.append(('- ' if coeff < 0 else '+ ') + term)

        power -= 1

    return parts


def format_parts(conv):
    """The relation as the pieces it may be broken between.

    A formula is too long for one line, and none of its terms may be broken
    across two - not by hard spaces alone, at any rate, as a browser will
    happily break after the closing parenthesis of (g - r)^2 as well. So the
    pieces are kept apart here and the page renders each of them as its own
    unbreakable span; the only place a line may end is between two of them.
    """
    parts = ['%s = %s' % (conv['output'], conv['base'])]

    for term in conv['terms']:
        parts += format_term(term['color'], term['coeffs'])

    if conv['offset']:
        parts.append(('- ' if conv['offset'] < 0 else '+ ') + '%.5g' % abs(conv['offset']))

    return parts


def get_conversions():
    """The relations, in the sections the page shows them in."""
    groups = OrderedDict()

    for gid, group in GROUPS.items():
        groups[gid] = dict(group, id=gid, conversions=[])

    for conv in CONVERSIONS:
        conv = dict(conv, formula=format_parts(conv))
        # Every distinct colour the calculator will need a field for
        conv['colors'] = [_['color'] for _ in conv['terms']]
        groups[conv['group']]['conversions'].append(conv)

    return list(groups.values())


def passbands(request):
    """The photometric conversions, documented and calculable."""
    groups = get_conversions()

    context = {
        'groups': groups,
        # The same relations, for the calculator in the page to evaluate
        'conversions': [conv for group in groups for conv in group['conversions']],
    }

    return TemplateResponse(request, 'passbands.html', context=context)
