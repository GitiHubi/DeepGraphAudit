# import class libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

mpl.rcParams['agg.path.chunksize'] = 1000000
plt.rcParams['agg.path.chunksize'] = 1000000

class VisualizationHandler(object):

    # define plain class constructor
    def __init__(self):

        # set plotting directory
        self.plot_dir = ''

    # set plot dir
    def set_plot_dir(self, plot_dir):

        # set plotting directory
        self.plot_dir = plot_dir

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d(self, data, z1_col_name, z2_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        sns.set_palette("tab20")

        # init subplot
        fig, ax1 = plt.subplots(1, 1)

        # scatter plot of embeddings
        ax1.scatter(data[z1_col_name], data[z2_col_name], marker='o', edgecolors='w', s=14, linewidth=0.1)

        #for i, label in enumerate(data.index):
        #ax1.text(data.iloc[i]['z1'], data.iloc[i]['z2'], label, fontsize=4)

        # set axis labels
        ax1.set_xlabel('[$z_1$]', fontsize=16)
        ax1.set_ylabel('[$z_2$]', fontsize=16)

        # set tick fontsize
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # set plot header
        ax1.set_title(title, fontsize=14)

        # set grid and tight plotting layout
        plt.grid(True)
        plt.tight_layout()

        # save plot to plotting directory
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_interval(self, data, z1_col_name, z2_col_name, c_col_name, filename, title, xlim=[15, 25], ylim=[-7.5, -12.5]):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        cm = plt.cm.get_cmap('RdYlBu')

        # init subplot
        fig, ax1 = plt.subplots(1, 1)

        # determine mean and std of reconstruction error
        mean = data[c_col_name].mean()
        std = data[c_col_name].std()

        # scatter plot of embeddings
        plot = ax1.scatter(data[z1_col_name], data[z2_col_name], c=data[c_col_name], vmin=mean-0.1*std, vmax=mean+0.1*std, marker='o', edgecolors='w', s=14, linewidth=0.1, cmap=cm)

        #for i, label in enumerate(data.index):
        #ax1.text(data.iloc[i]['z1'], data.iloc[i]['z2'], label, fontsize=4)

        # set axis labels
        ax1.set_xlabel('[$z_1$]', fontsize=16)
        ax1.set_ylabel('[$z_2$]', fontsize=16)

        # set axis limitations
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        # set tick fontsize
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # set plot header
        ax1.set_title(title, fontsize=14)

        # set scatter plot colorbar
        plt.colorbar(plot)

        # set grid and tight plotting layout
        plt.grid(True)
        plt.tight_layout()

        # save plot to plotting directory
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

        # close plot
        plt.close()

        # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_interactive(self, data, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        cm = plt.cm.get_cmap('RdYlBu')

        # determine mean and std of reconstruction error
        mean = data[c_col_name].mean()
        std = data[c_col_name].std()

        # scatter plot of embeddings
        fig = px.scatter(data
                  , x=z1_col_name
                  , y=z2_col_name
                  , color=c_col_name
                  , labels={z1_col_name : '<b>[z1]</b>', z2_col_name : '<b>[z2]</b>', c_col_name : '<b>Rec. Error</b>'}
                  , hover_data=['Y_JE_LINE_ITEMS', 'Y_PREPARER_ID', 'Y_SOURCE', 'Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NAME']
                  , title=title)

        # update axis layout
        fig.update_layout(
            title=dict(font_size=28, x=0.5, xanchor='center')
            , hoverlabel=dict(bgcolor="white", font_size=14)
            , xaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            , yaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            )

        # update coloraxis layout
        fig.update_coloraxes(colorbar_tickfont_size=18, cmin=mean-0.1*std, cmax=mean+0.1*std)

        # save plot to plotting directory
        fig.write_html(os.path.join(self.plot_dir, filename))

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_attribute(self, data, attribute, z1_col_name, z2_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        sns.set_palette("tab20")

        # init subplot
        fig, ax1 = plt.subplots(1, 1)

        # determine feature groups
        attribute_values = data.groupby(attribute)

        # iterate over feature groups
        for attribute_name, attribute_value in attribute_values:

            # scatter plot of embeddings
            ax1.scatter(attribute_value[z1_col_name], attribute_value[z2_col_name], marker='o', edgecolors='w', s=14, linewidth=0.1, label=str(attribute_name))

        for i, label in enumerate(data.index):
            ax1.text(data.iloc[i]['z1'], data.iloc[i]['z2'], label, fontsize=4)

        # set axis labels
        ax1.set_xlabel('[$z_1$]', fontsize=16)
        ax1.set_ylabel('[$z_2$]', fontsize=16)

        # set tick fontsize
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # set plot header
        ax1.set_title(title, fontsize=14)

        # set grid and tight plotting layout
        plt.grid(True)
        plt.tight_layout()

        # save plot to plotting directory
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

        # close plot
        plt.close()
