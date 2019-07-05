#!/usr/bin/env python3

"""Script to run LOW station beam error simulations."""

from __future__ import print_function, division
import json
import logging
import os
import sys

from astropy.io import fits
from astropy.time import Time, TimeDelta
import matplotlib
matplotlib.use('Agg')
# pylint: disable=wrong-import-position
from matplotlib import pyplot as plt
import numpy
import oskar


LOG = logging.getLogger()


def make_vis_data(settings, sky, tel):
    """Run simulation using supplied settings."""
    out_ms = settings['interferometer/ms_filename']
    out_vis = settings['interferometer/oskar_vis_filename']
    out_files = []
    if out_ms and not os.path.exists(out_ms):
        out_files.append(out_ms)
    if out_vis and not os.path.exists(out_vis):
        out_files.append(out_vis)
    if not out_files:
        LOG.info("Skipping simulation, as output data already exist.")
        return

    LOG.info("Simulating %s", ', '.join(out_files))
    if sky.num_sources == 1:
        settings['simulator/use_gpus'] = 'false'
        settings['simulator/max_sources_per_chunk'] = '2'
    #print(json.dumps(settings.to_dict(), indent=4))
    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(sky)
    sim.set_telescope_model(tel)
    sim.run()


def make_diff_image_stats(filename1, filename2, out_image_root=None):
    """Make an image of the difference between two visibility data sets.

    This function assumes that the observation parameters for both data sets
    are identical. (It will fail horribly otherwise!)
    """
    # Set up an imager.
    imager = oskar.Imager(precision='double')
    imager.set(fov_deg=5.0, image_size=5000, fft_on_gpu=True)
    imager.set(algorithm='W-projection')
    if out_image_root is not None:
        imager.output_root = out_image_root

    LOG.info("Imaging differences between '%s' and '%s'", filename1, filename2)
    (hdr1, handle1) = oskar.VisHeader.read(filename1)
    (hdr2, handle2) = oskar.VisHeader.read(filename2)
    block1 = oskar.VisBlock.create_from_header(hdr1)
    block2 = oskar.VisBlock.create_from_header(hdr2)
    if hdr1.num_blocks != hdr2.num_blocks or \
            hdr1.num_stations != hdr2.num_stations or \
            hdr1.max_times_per_block != hdr2.max_times_per_block or \
            hdr1.max_channels_per_block != hdr2.max_channels_per_block:
        raise RuntimeError("'%s' and '%s' have different dimensions!" %
                           (filename1, filename2))
    imager.coords_only = True
    for i_block in range(hdr1.num_blocks):
        block1.read(hdr1, handle1, i_block)
        imager.update_from_block(hdr1, block1)
    imager.coords_only = False
    imager.check_init()
    LOG.info("Using %d W-planes", imager.num_w_planes)
    for i_block in range(hdr1.num_blocks):
        block1.read(hdr1, handle1, i_block)
        block2.read(hdr2, handle2, i_block)
        block1.cross_correlations()[...] -= block2.cross_correlations()
        imager.update_from_block(hdr1, block1)
    del handle1, handle2, hdr1, hdr2, block1, block2

    # Finalise image and return it to Python.
    output = imager.finalise(return_images=1)
    image = output['images'][0]

    LOG.info("Generating image statistics")
    image_size = imager.image_size
    box_size = int(0.1 * image_size)
    centre = image[
        (image_size - box_size)//2:(image_size + box_size)//2,
        (image_size - box_size)//2:(image_size + box_size)//2]
    return {
        'image_max': numpy.max(image),
        'image_min': numpy.min(image),
        'image_maxabs': numpy.max(numpy.abs(image)),
        'image_medianabs': numpy.median(numpy.abs(image)),
        'image_mean': numpy.mean(image),
        'image_rms': numpy.sqrt(numpy.mean(image**2)),
        'image_centre_maxabs': numpy.max(numpy.abs(centre)),
        'image_centre_mean': numpy.mean(centre),
        'image_centre_std': numpy.std(centre),
        'image_centre_rms': numpy.sqrt(numpy.mean(centre**2))
    }


def run_single(prefix_field, settings, sky, tel,
               gain_std_dB, phase_std_deg, obs_length, out0_name, results):
    """Run a single simulation and generate image statistics for it."""
    out = '%s_%d_sec_%.3f_dB_%.2f_deg' % (
        prefix_field, obs_length, gain_std_dB, phase_std_deg)
    if out in results:
        LOG.info("Using cached results for '%s'", out)
        return
    out_name = out + '.vis'
    gain_std = numpy.power(10.0, gain_std_dB / 20.0) - 1.0
    tel.override_element_gains(1.0, gain_std)
    tel.override_element_phases(phase_std_deg)
    settings['interferometer/oskar_vis_filename'] = out_name
    make_vis_data(settings, sky, tel)
    out_image_root = None
    results[out] = make_diff_image_stats(out0_name, out_name, out_image_root)
    #os.remove(out_name)  # Delete visibility data to save space.


def make_sky_model(sky0, settings, radius_deg, flux_min_outer_jy):
    """Filter sky model.

    Includes all sources within the given radius, and sources above the
    specified flux outside this radius.
    """
    # Get pointing centre.
    ra0_deg = float(settings['observation/phase_centre_ra_deg'])
    dec0_deg = float(settings['observation/phase_centre_dec_deg'])

    # Create "inner" and "outer" sky models.
    sky_inner = sky0.create_copy()
    sky_outer = sky0.create_copy()
    sky_inner.filter_by_radius(0.0, radius_deg, ra0_deg, dec0_deg)
    sky_outer.filter_by_radius(radius_deg, 180.0, ra0_deg, dec0_deg)
    sky_outer.filter_by_flux(flux_min_outer_jy, 1e9)
    LOG.info("Number of sources in sky0: %d", sky0.num_sources)
    LOG.info("Number of sources in inner sky model: %d", sky_inner.num_sources)
    LOG.info("Number of sources in outer sky model above %.3f Jy: %d",
             flux_min_outer_jy, sky_outer.num_sources)
    sky_outer.append(sky_inner)
    LOG.info("Number of sources in output sky model: %d", sky_outer.num_sources)
    return sky_outer


def get_start_time(ra0_deg, length_sec):
    """Returns optimal start time for field RA and observation length."""
    t = Time('2000-01-01 00:00:00', scale='utc', location=('116.764d', '0d'))
    dt_hours = 24.0 - t.sidereal_time('apparent').hour + (ra0_deg / 15.0)
    start = t + TimeDelta(dt_hours * 3600.0 - length_sec / 2.0, format='sec')
    return start.value


def make_plot(prefix, field_name, metric_key, results, axis_length,
              axis_gain, axis_phase, single_source_offset):
    """Plot selected results."""

    # Axis setup.
    ax1 = plt.subplot(111)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    for gain in axis_gain:
        for phase in axis_phase:
            # Get data for line plot.
            X = numpy.array(axis_length)
            Y = numpy.zeros(X.shape)
            label = '%.3f_dB_%.2f_deg' % (gain, phase)
            for x, y in numpy.nditer([X, Y], op_flags=['readwrite']):
                key = '%s_%s_%d_sec_%.3f_dB_%.2f_deg' % (
                    prefix, field_name, x, gain, phase)
                if key in results:
                    y[...] = results[key][metric_key]

            # Line plot.
            plt.plot(X, Y, marker='o', label=label)

    # Title and axis labels.
    metric_name = '[ UNKNOWN ]'
    if metric_key == 'image_centre_rms':
        metric_name = 'Central RMS [Jy/beam]'
    elif metric_key == 'image_maxabs':
        metric_name = 'MAX(ABS(image)) [Jy/beam]'
    elif metric_key == 'image_medianabs':
        metric_name = 'MEDIAN(ABS(image)) [Jy/beam]'
    sky_model = prefix
    if single_source_offset >= 0:
        sky_model = 'single source'
    plt.title('%s for %s field (%s)' % (metric_name, field_name, sky_model))
    plt.xlabel('Observation length [sec]')
    plt.ylabel('%s' % metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s_%s_%s.png' % (prefix, field_name, metric_key))
    plt.close('all')


def run_set(sky0, prefix, base_settings, fields, axis_length,
            axis_gain, axis_phase, single_source_offset, plot_only):
    """Runs a set of simulations."""
    if not plot_only:
        # Load base telescope model.
        settings = oskar.SettingsTree('oskar_sim_interferometer')
        settings.from_dict(base_settings)
        tel = oskar.Telescope(settings=settings)

    # Iterate over fields.
    for field_name, field in fields.items():
        # Load result set, if it exists.
        prefix_field = prefix + '_' + field_name
        results = {}
        json_file = prefix_field + '_results.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as input_file:
                results = json.load(input_file)

        if not plot_only:
            # Update settings for field.
            settings_dict = base_settings.copy()
            settings_dict.update(field)
            settings.from_dict(settings_dict)
            ra_deg = float(settings['observation/phase_centre_ra_deg'])
            dec_deg = float(settings['observation/phase_centre_dec_deg'])
            tel.set_phase_centre(ra_deg, dec_deg)

            # Create the sky model.
            if single_source_offset >= 0:
                sky = oskar.Sky.from_array(
                    [ra_deg, dec_deg - single_source_offset, 1.0])
            else:
                sky = make_sky_model(sky0, settings, 20.0, 10.0)

            # Iterate over observation lengths.
            for length in axis_length:
                num_times = int(numpy.round(length / 60.0))
                settings['observation/length'] = str(length)
                settings['observation/num_time_steps'] = str(num_times)
                settings['observation/start_time_utc'] = get_start_time(
                    ra_deg, length)

                # Simulate the 'perfect' case.
                tel.override_element_gains(1.0, 0.0)
                tel.override_element_phases(0.0)
                out0_name = '%s_%d_sec_no_errors.vis' % (prefix_field, length)
                settings['interferometer/oskar_vis_filename'] = out0_name
                make_vis_data(settings, sky, tel)

                # Simulate the error cases.
                for gain_std_dB in axis_gain:
                    for phase_std_deg in axis_phase:
                        run_single(prefix_field, settings, sky, tel,
                                   gain_std_dB, phase_std_deg, length,
                                   out0_name, results)

        # Generate plot for the field.
        if single_source_offset >= 0:
            make_plot(prefix, field_name, 'image_centre_rms', results,
                      axis_length, axis_gain, axis_phase, single_source_offset)
        else:
            make_plot(prefix, field_name, 'image_maxabs', results,
                      axis_length, axis_gain, axis_phase, single_source_offset)
            make_plot(prefix, field_name, 'image_medianabs', results,
                      axis_length, axis_gain, axis_phase, single_source_offset)

        # Save result set.
        with open(json_file, 'w') as output_file:
            json.dump(results, output_file, indent=4)


def main():
    """Main function."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)

    # Define common settings.
    base_settings = {
        'simulator': {
            'double_precision': 'true',
            'use_gpus': 'true',
            'max_sources_per_chunk': '23000'
        },
        'observation' : {
            'start_frequency_hz': '100e6',
            'frequency_inc_hz': '100e3'
        },
        'telescope': {
            'input_directory': 'SKA1-LOW_SKO-0000422_Rev3_38m.tm',
            'pol_mode': 'Full',
            'aperture_array/array_pattern/element/gain': '1.0'
        },
        'interferometer': {
            'channel_bandwidth_hz': '100e3',
            'time_average_sec': '1.0',
            'max_time_samples_per_block': '4',
            'ignore_w_components': 'false'
        }
    }

    # Define axes of parameter space.
    axis_length = [60, 600, 1800, 3600, 7200, 14400]
    axis_gain = [0.27]
    axis_phase = [1.82]
    fields = {
        'EoR0': {
            'observation/phase_centre_ra_deg': '0.0',
            'observation/phase_centre_dec_deg': '-27.0'
        }
        # 'EoR1': {
        #     'observation/phase_centre_ra_deg': '60.0',
        #     'observation/phase_centre_dec_deg': '-30.0'
        # },
        # 'EoR2': {
        #     'observation/phase_centre_ra_deg': '170.0',
        #     'observation/phase_centre_dec_deg': '-10.0'
        # }
    }

    # Load GLEAM catalogue from FITS binary table.
    hdulist = fits.open('GLEAM_EGC.fits')
    # pylint: disable=no-member
    cols = hdulist[1].data[0].array
    data = numpy.column_stack(
        (cols['RAJ2000'], cols['DEJ2000'], cols['peak_flux_wide']))
    data = data[data[:, 2].argsort()[::-1]]
    sky0 = oskar.Sky.from_array(data)

    # Plot histogram of source flux values.
    min_flux = numpy.min(data[:, 2])
    max_flux = numpy.max(data[:, 2])
    bins = numpy.logspace(numpy.log10(min_flux), numpy.log10(max_flux), 50)
    plt.hist(data[:, 2], bins=bins)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('Log10(Flux [Jy])')
    plt.ylabel('Log10(source count)')
    plt.savefig('GLEAM_flux_histogram.png')
    plt.close()

    # A list of bright sources.
    # Sgr A: guesstimates only!
    # For A: data from the Molonglo Southern 4 Jy sample (VizieR).
    # Others from GLEAM reference paper, Hurley-Walker et al. (2017), Table 2.
    # pylint: disable=bad-whitespace
    sky_bright = oskar.Sky.from_array(numpy.array((
        [266.41683, -29.00781,  2000,0,0,0,   0,    0,    0, 3600, 3600, 0],
        [ 50.67375, -37.20833,   528,0,0,0, 178e6, -0.51, 0, 0, 0, 0],  # For
        [201.36667, -43.01917,  1370,0,0,0, 200e6, -0.50, 0, 0, 0, 0],  # Cen
        [139.52500, -12.09556,   280,0,0,0, 200e6, -0.96, 0, 0, 0, 0],  # Hyd
        [ 79.95833, -45.77889,   390,0,0,0, 200e6, -0.99, 0, 0, 0, 0],  # Pic
        [252.78333,   4.99250,   377,0,0,0, 200e6, -1.07, 0, 0, 0, 0],  # Her
        [187.70417,  12.39111,   861,0,0,0, 200e6, -0.86, 0, 0, 0, 0],  # Vir
        [ 83.63333,  22.01444,  1340,0,0,0, 200e6, -0.22, 0, 0, 0, 0],  # Tau
        [299.86667,  40.73389,  7920,0,0,0, 200e6, -0.78, 0, 0, 0, 0],  # Cyg
        [350.86667,  58.81167, 11900,0,0,0, 200e6, -0.41, 0, 0, 0, 0]   # Cas
        )))
    sky0.append(sky_bright)

    # Run simulations.
    run_set(sky0, 'GLEAM_A-team', base_settings, fields, axis_length,
            axis_gain, axis_phase, -1.0, False)

if __name__ == '__main__':
    main()
