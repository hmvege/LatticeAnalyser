from post_analysis.core.postcore import PostCore
from tools.folderreadingtools import check_folder
from tools.latticefunctions import get_lattice_spacing
from tqdm import tqdm
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
import subprocess
import os
import types
import re

class FlowGif(PostCore):
	y_limits = None
	x_limits = None

	def image_creator(self, euclidean_percent=None, gif_betas="all", 
		plot_together=False, atype=None, error_shape="band", dpi=200):
		"""
		Creates a folder for storing the smearing gif in.

		Args:
			euclidean_percent: optional, float, what euclidean time percent we are
				creating the gif at.
			gif_betas: optional, str or list. Which betas to plot. Options:
				"all": plots all beta values available in analysis.
				list of floats: creates gif of beta values in list, 
					e.g. [6.0, 6.1].
			plot_together: optional, bool. If true, will plot different beta 
				values together in the same gif.
			atype: optional, str, analysis type. Default is bootstrap.
			error_shape: optional, str, plot type. Choices: band, bars.
				Default is band.
			dpi: int, optional. Default is 200.

		Raises:
			AssertionError if gif_betas is not of type string or list of floats.
		"""
		print "Creating gif for %s" % self.observable_name_compact

		self.plot_together = plot_together
		self.error_shape = error_shape
		self.dpi = dpi

		# In case we are looking at plots with no options for selecting a 
		# euclidean time slice.
		self.subsubdim = False
		if not isinstance(euclidean_percent, types.NoneType):
			self.tEucl = {
				beta: int((self.lattice_sizes[beta][-1]-1)*euclidean_percent) \
					for beta in self.beta_values
			}
			self.subsubdim = True

		if isinstance(atype, types.NoneType):
			atype = "bootstrap"

		assert isinstance(gif_betas, (str, list)), \
			"gif_betas is not of type list or str: %s" % type(gif_betas)

		if gif_betas == "all":
			self.gif_betas = sorted(self.beta_values)
		else:
			assert np.all([isinstance(b, float) for b in gif_betas]), \
				"not all beta values are floats: %s" % \
					", ".join([str(i) for i in gif_betas])
			self.gif_betas = sorted(gif_betas)

		self.gif_folder = os.path.join(self.output_folder_path, "gif")
		check_folder(self.gif_folder, dryrun=self.dryrun, verbose=self.verbose)

		# Creates folder to store images in
		self.img_folder = os.path.join(self.gif_folder, "img")
		check_folder(self.img_folder, dryrun=self.dryrun, verbose=self.verbose)

		self.gif_plot_values = {}
		self.flow_times = []

		# Sets up gif image creation data
		for beta in self.beta_values:
			# Gets the flow from digit patterns of the shape #.####
			flow_times = sorted([float(re.findall(r"(\d{1}\.\d{4})", i)[0]) \
				for i in self.data[atype][beta].keys()])

			# Sets up plot values
			if self.subsubdim:
				self.gif_plot_values[beta] = {
					ftime: self.data[atype][beta]["tflow%.4f" % ftime] \
						["te%04d" % self.tEucl[beta]] for ftime in flow_times
				}
			else:
				self.gif_plot_values[beta] = {
					ftime: self.data[atype][beta]["tflow%.4f" % ftime] \
						for ftime in flow_times
				}

			# Scales x-axis values
			for ftime in flow_times:
				self.gif_plot_values[beta][ftime]["x"] *= \
					get_lattice_spacing(beta)

			self.flow_times.append(flow_times)
		# Sets a single flow time to be referenced
		assert np.all([np.allclose(i, [i[0]]) for i in np.array(self.flow_times).T]), \
			"flow times created for each beta to not match."
		self.flow_times = self.flow_times[0]

		# Offsets x-axis values so that we are centered correctly on the 
		# smearing point.
		if self.subsubdim:
			max_val = []
			for beta in self.beta_values:
				max_val.append(self.gif_plot_values[beta][self.flow_times[0]] \
					["x"][self.tEucl[beta]])

			for i, beta in enumerate(self.beta_values[:-1]):
				offset = max_val[-1] - max_val[i]
				for ftime in self.flow_times:
					self.gif_plot_values[beta][ftime]["x"] += offset

		if self.plot_together:
			self.plot_values = {}
			self.plot_betas = self.gif_betas

			# Plots all beta values together
			for ftime in tqdm(self.flow_times, "Plotting betas together"):
				self._plot_gif_image(ftime)

			self.create_gif()

			# Removes figures from folder
			self.clean_up()
		else:
			for beta in self.gif_betas:
				self.plot_values = {}
				self.plot_betas = [beta]

				tqdm_msg = "Plotting for beta=%.2f" % beta
				for ftime in tqdm(self.flow_times, tqdm_msg):
					self._plot_gif_image(ftime)

				# Converts to gif PASS ON FIGURE LIST?
				self.create_gif()

				# Removes figures from folder
				self.clean_up()

	def _plot_gif_image(self, flow_time):
		"""
		Plots a single frame for gif.

		Args:
			flow_time: float, key to be used in retrieving data.
		"""

		fig = plt.figure(dpi=self.dpi)
		ax = fig.add_subplot(111)

		# Retrieves values to plot
		for beta in self.plot_betas:
			value = self.gif_plot_values[beta][flow_time]
			x = value["x"]
			y = value["y"]
			y_err = value["y_error"]

			label_str = r"$\beta = %.2f$" % beta

			if self.error_shape == "band":
				ax.plot(x, y, "-", label=label_str, color=self.colors[beta])
				ax.fill_between(x, y - y_err, y + y_err, alpha=0.5, 
					edgecolor='', facecolor=self.colors[beta])
			elif self.error_shape == "bars":
				ax.errorbar(x, y, yerr=y_err, capsize=5, fmt="_", ls=":", 
					label=label_str, color=self.colors[beta],
					ecolor=self.colors[beta])
			else:
				raise KeyError("%s not a recognized plot type" % self.error_shape)

		# Sets the title string
		title_string = r"$\sqrt{8t_f} = %2.4f[fm]$" % flow_time

		# Basic plotting commands
		ax.grid(True)
		ax.set_title(title_string)
		ax.set_xlabel(self.x_label)
		ax.set_ylabel(self.y_label)
		ax.legend(loc="lower right", prop={"size": 8})

		# Sets axes limits if provided
		if not isinstance(self.x_limits, types.NoneType):
			ax.set_ylim(self.x_limits)
		if not isinstance(self.y_limits, types.NoneType):
			ax.set_ylim(self.y_limits)

		# Saves and closes figure
		fname = os.path.join(self.img_folder, 
			"img_%s.png" % str("%0.4f" % flow_time).replace(".", ""))

		plt.savefig(fname)
		plt.close(fig)

	def create_gif(self):
		"""Creates a gif from images in gif the gif folder."""

		gif_path = os.path.join(self.gif_folder,
			"%s_%s.gif" % (self.observable_name_compact,
				"_".join(["b%.2f" % b for b in self.plot_betas])))

		fig_base_name = "img_*.png"
		input_paths = os.path.join(self.img_folder, fig_base_name)

		cmd = ['convert', '-delay', '10', '-loop', '0', input_paths,
			gif_path]

		print "> %s" % " ".join(cmd)	
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		read_out = proc.stdout.read()
		print read_out
		print "\nGif creation done.\n"

	def clean_up(self):
		"""Removes images in img folder."""
		for img in os.listdir(self.img_folder):
			img_path = os.path.join(self.img_folder, img)
			if not self.dryrun:
				os.remove(img_path)
			if self.verbose:
				print "rm >%s" % img_path
		if self.verbose:
			print "Cleaned up folder %s" % self.img_folder

if __name__ == '__main__':
	exit("Import from post analysis child.")