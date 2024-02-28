# import python libraries
import os
import numpy as np
import pandas as pd

# import networkx
import networkx as nx

# import matplotlib libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# import plotly libraries
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

mpl.rcParams['agg.path.chunksize'] = 1000000
plt.rcParams['agg.path.chunksize'] = 1000000

# class VisualisationHandler
class VisualisationHandler(object):

    # define plain class constructor
    def __init__(self):

        pass

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d(self, parameter, data, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        sns.set_palette("tab10")

        # init subplot
        fig, ax1 = plt.subplots(1, 1)

        # determine feature groups
        attribute_values = data.groupby(c_col_name)
        color_map = {'regular': 'C0', 'global': 'C1', 'local': 'C3'}
        marker_map = {'regular': 'o', 'global': '^', 'local': '*'}
        zorder = {'regular': 1, 'global': 2, 'local': 3}
        size = {'regular': 12, 'global': 30, 'local': 30}

        # iterate over feature groups
        for attribute_name, attribute_value in attribute_values:

            # scatter plot of embeddings
            ax1.scatter(attribute_value[z1_col_name], attribute_value[z2_col_name], c=color_map[attribute_name], label=attribute_name, zorder=zorder[attribute_name], marker=marker_map[attribute_name],  s=size[attribute_name], edgecolors='w', linewidth=0.1)

        # set axis labels
        ax1.set_xlabel('[$z_1$]', fontsize=16)
        ax1.set_ylabel('[$z_2$]', fontsize=16)

        # set tick fontsize
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # set legend
        ax1.legend(fontsize=12, loc='upper right', fancybox=True, framealpha=0.5)

        # set plot header
        ax1.set_title(title, fontsize=12)

        # set grid and tight plotting layout
        plt.grid(True)
        plt.tight_layout()

        # save plot to plotting directory
        plt.savefig(os.path.join(parameter['vis_sub_dir'], filename), dpi=300)

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_error(self, parameter, data, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        cm = plt.cm.get_cmap('coolwarm')

        # determine colorbar start and end range
        range_end = data[c_col_name].mean() + 0.01 * data[c_col_name].std()

        # init subplot
        fig, ax1 = plt.subplots(1, 1)

        # scatter plot of embeddings
        plot = ax1.scatter(data[z1_col_name], data[z2_col_name], c=data[c_col_name], vmin=0.0, vmax=range_end, marker='o', edgecolors='w', s=14, linewidth=0.1, cmap=cm)

        # set axis labels
        ax1.set_xlabel('[$z_1$]', fontsize=16)
        ax1.set_ylabel('[$z_2$]', fontsize=16)

        # set tick fontsize
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # set plot header
        ax1.set_title(title, fontsize=12)

        # set scatter plot colorbar
        plt.colorbar(plot)

        # set grid and tight plotting layout
        plt.grid(True)
        plt.tight_layout()

        # save plot to plotting directory
        plt.savefig(os.path.join(parameter['vis_sub_dir'], filename), dpi=300)

        # close plot
        plt.close()

        # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_anomalies(self, parameter, data, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        sns.set_palette("tab10")

        # init subplot
        fig, ax1 = plt.subplots(1, 1)

        # determine feature groups
        attribute_values = data.groupby(c_col_name)
        color_map = {'0': 'C0', '1': 'C3'}   #color_map = {'1': 'C0', '-1': 'C3'}
        marker_map = {'0': 'o', '1': '*'}   #marker_map = {'1': 'o', '-1': '*'}
        zorder = {'0': 1, '1': 2}          #zorder = {'1': 1, '-1': 2}
        size = {'0': 12, '1': 30}         #size = {'1': 12, '-1': 30}
        label = {'0': 'regular', '1': 'anomaly'}    # label = {'1': 'regular', '-1': 'anomaly'}

        # iterate over feature groups
        for attribute_name, attribute_value in attribute_values:

            # scatter plot of embeddings
            ax1.scatter(attribute_value[z1_col_name], attribute_value[z2_col_name], c=color_map[str(attribute_name)], label=label[str(attribute_name)], zorder=zorder[str(attribute_name)], marker=marker_map[str(attribute_name)],  s=size[str(attribute_name)], edgecolors='w', linewidth=0.1)

        # set axis labels
        ax1.set_xlabel('[$z_1$]', fontsize=16)
        ax1.set_ylabel('[$z_2$]', fontsize=16)

        # set tick fontsize
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # set legend
        ax1.legend(fontsize=12, loc='upper right', fancybox=True, framealpha=0.5)

        # set plot header
        ax1.set_title(title, fontsize=12)

        # set grid and tight plotting layout
        plt.grid(True)
        plt.tight_layout()

        # save plot to plotting directory
        plt.savefig(os.path.join(parameter['vis_sub_dir'], filename), dpi=300)

        # close plot
        plt.close()

    # plot single journal entry graph
    def plot_single_journal_entry_graph(self, parameter, entry_graph, filename):

        # define graph drawing options
        nodes_draw_options = {
            'node_color': 'orange'
            , 'alpha': 0.8
            , 'node_size': 1200
        }

        # define graph drawing options
        edges_draw_options = {
            'edge_color': 'gray'
            , 'width': 2
            , 'alpha': 0.6
            , 'arrowstyle': '-|>'
        }

        # set graph visualization layout
        pos_nodes = nx.spring_layout(entry_graph)

        # draw network nodes
        nx.draw_networkx_nodes(entry_graph, pos_nodes, **nodes_draw_options)

        # determine node attributes
        node_attributes = nx.get_node_attributes(entry_graph, 'account')

        # customize node attributes
        custom_node_labels = {}
        for node, attribute in node_attributes.items():
            custom_node_labels[node] = str(node)

        # draw network node labels
        nx.draw_networkx_labels(entry_graph, pos_nodes, font_color='black', font_size=16, alpha=1.0)

        # init and compute attribute positions
        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0] + 0.08, coords[1] + 0.08)

        # customize node attributes
        custom_node_attributes = {}
        for node, attribute in node_attributes.items():
            custom_node_attributes[node] = attribute.values[0]

        # draw graph network labels
        nx.draw_networkx_labels(entry_graph, pos_attrs, labels=custom_node_attributes, font_size=10, alpha=0.7)

        # draw network edges
        nx.draw_networkx_edges(entry_graph, pos_nodes, arrows=True, **edges_draw_options)

        # remove black box around the plot
        plt.box(False)

        # save plot to plotting directory
        plt.savefig(os.path.join(parameter['gra_sub_dir'], filename), dpi=100)

        # close plot
        plt.close()

    # plot single journal entry graph
    def plot_entire_journal_entry_graph(self, parameter, entry_graph, pos_nodes, filename):

        # define graph drawing options
        nodes_draw_options = {
            'node_color': 'orange'
            , 'alpha': 0.8
            , 'node_size': 120
        }

        # define graph drawing options
        edges_draw_options = {
            'edge_color': 'gray'
            , 'width': 1
            , 'alpha': 0.6
            , 'arrowstyle': '-|>'
        }

        # draw network nodes
        nx.draw_networkx_nodes(entry_graph, pos_nodes, **nodes_draw_options)

        # determine node attributes
        node_attributes = nx.get_node_attributes(entry_graph, 'account')

        # customize node attributes
        custom_node_labels = {}
        for node, attribute in node_attributes.items():
            custom_node_labels[node] = str(node)

        # draw network node labels
        nx.draw_networkx_labels(entry_graph, pos_nodes, font_color='black', font_size=4, alpha=1.0)

        # init and compute attribute positions
        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0] + 0.08, coords[1] + 0.08)

        # customize node attributes
        custom_node_attributes = {}
        for node, attribute in node_attributes.items():
            custom_node_attributes[node] = attribute.values[0]

        # draw graph network labels
        # nx.draw_networkx_labels(entry_graph, pos_attrs, labels=custom_node_attributes, font_size=5, alpha=0.7)

        # draw network edges
        nx.draw_networkx_edges(entry_graph, pos_nodes, arrows=True, **edges_draw_options)

        # remove black box around the plot
        plt.box(False)

        # save plot to plotting directory
        plt.savefig(os.path.join(parameter['gra_sub_dir'], filename), dpi=200)

        # close plot
        plt.close()


    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_interval(self, data, z1_col_name, z2_col_name, c_col_name, filename, title, xlim=[15, 25], ylim=[-7.5, -12.5]):

        # set plotting appearance
        plt.style.use('seaborn')
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
        #plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_sap_embeddings_2d_interactive(self, data, attributes, hover, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        # cm = plt.cm.get_cmap('RdYlBu')

        # set the sub-plots grid layout
        specs=[[{'colspan': len(attributes)}, None, None, None, None, None, None, None, None],
               [{}, {}, {}, {}, {}, {}, {}, {}, {}],
               ]

        # set the sub-plots titles
        titles=[title] + ['Attribute:<br><b>{}</b>'.format(attribute) for attribute in attributes]

        # init subplots
        fig = make_subplots(rows=2, cols=len(attributes), specs=specs, subplot_titles=titles, vertical_spacing=0.1)

        # prepare hover data
        hover=np.stack((data['Y_BELNR']
                        , data['Y_BUZEI']
                        , data['Y_BLART']
                        , data['Y_USNAM']
                        , data['Y_TCODE']
                        , data['Y_BSCHL']
                        , data['Y_HKONT_TEXT']
                        , data['Y_PRCTR']
                        , data['Y_KOSTL']
                        , data['Y_DMBTR']
                        , np.round(data[c_col_name], 4)
                        ), axis=-1)

        # prepare hover template
        template='<b>Y_BELNR</b>: %{customdata[0]}<br>' \
                 '<b>Y_BUZEI</b>: %{customdata[1]}<br>' \
                 '<b>Y_BLART</b>: %{customdata[2]}<br>' \
                 '<b>Y_USNAM</b>: %{customdata[3]}<br>' \
                 '<b>Y_TCODE</b>: %{customdata[4]}<br>' \
                 '<b>Y_BSCHL</b>: %{customdata[5]}<br>' \
                 '<b>Y_HKONT</b>: %{customdata[6]}<br>' \
                 '<b>Y_PRCTR</b>: %{customdata[7]}<br>' \
                 '<b>Y_KOSTL</b>: %{customdata[8]}<br>' \
                 '<b>Y_DMBTR</b>: %{customdata[9]}<br>' \
                 '<b>Y_REC_ERROR</b>: %{customdata[10]}<br>' \
                 '<extra></extra>'

        # add trace to figure
        fig.add_trace(

            # create single scatter plot
            go.Scatter(x=data[z1_col_name], y=data[z2_col_name], mode='markers', marker_color=data[c_col_name], marker=dict(showscale=True, colorbar={'title': '<b>Rec. Error</b>'}), customdata=hover, showlegend=False, hovertemplate=template, opacity=0.7),

            # add scatter plot position
            row=1, col=1

        )

        # set x-axis and y-axis label
        fig['layout']['xaxis{}'.format(str(1))]['title']='<b>[z1]</b>'
        fig['layout']['yaxis{}'.format(str(1))]['title']='<b>[z2]</b>'

        # set colorbar title
        fig['layout']['coloraxis']['colorbar']['title'] = '<b>Rec. Error</b>'

        # iterate over distinct attributes
        for i, attribute in enumerate(attributes):

            # case: continuous attribute
            if attribute == 'Y_DMBTR':

                # prepare color codes
                colors = pd.Categorical(data[attribute]).codes

                # add trace to figure
                fig.add_trace(

                    # create single scatter plot
                    go.Scatter(x=data[z1_col_name], y=data[z2_col_name], mode='markers', marker_color=data[attribute], marker=dict(showscale=False), customdata=hover, showlegend=False, hovertemplate=template, opacity=0.7),

                    # add scatter plot position
                    row=2, col=i+1

                )

            # case: categorical attribute
            else:

                # prepare color codes
                colors = pd.Categorical(data[attribute]).codes

                # add trace to figure
                fig.add_trace(

                    # create single scatter plot
                    go.Scatter(x=data[z1_col_name], y=data[z2_col_name], mode='markers', marker=dict(color=colors, size=8), customdata=hover, showlegend=False, hovertemplate=template, opacity=0.7),

                    # add scatter plot position
                    row=2, col=i+1

                )

            # case: initial plot
            if i == 0:

                # set x-axis and y-axis label
                fig['layout']['xaxis{}'.format(str(i+2))]['title']='<b>[z1]</b>'
                fig['layout']['yaxis{}'.format(str(i+2))]['title']='<b>[z2]</b>'

            # case: non-initial plot
            else:

                # set x-axis label
                fig['layout']['xaxis{}'.format(str(i+2))]['title']='<b>[z1]</b>'

        # save plot to plotting directory
        #fig.write_html(os.path.join(self.plot_dir, filename))

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_ey_embeddings_2d_interactive(self, data, attributes, hover, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000

        # set the sub-plots grid layout
        specs=[[{'colspan': len(attributes)}, None, None, None, None, None, None],
               [{}, {}, {}, {}, {}, {}, {}],
               ]

        # set the sub-plots titles
        titles=[title] + ['Attribute:<br><b>{}</b>'.format(attribute) for attribute in attributes]

        # init subplots
        fig = make_subplots(rows=2, cols=len(attributes), specs=specs, subplot_titles=titles, vertical_spacing=0.1)

        # prepare hover data
        hover=np.stack((data['Y_JE_IDENTIFIER']
                        , data['Y_BUZEI']
                        , data['Y_PREPARER_ID']
                        , data['Y_SOURCE']
                        , data['Y_ACCOUNT_TYPE']
                        , data['Y_ACCOUNT_CLASS']
                        , data['Y_GL_ACCOUNT_NAME']
                        , data['Y_DMBTR']
                        , np.round(data[c_col_name], 4)
                        ), axis=-1)

        # prepare hover template
        template='<b>Y_JE_IDENTIFIER</b>: %{customdata[0]}<br>' \
                 '<b>Y_BUZEI</b>: %{customdata[1]}<br>' \
                 '<b>Y_PREPARER_ID</b>: %{customdata[2]}<br>' \
                 '<b>Y_SOURCE</b>: %{customdata[3]}<br>' \
                 '<b>Y_ACCOUNT_TYPE</b>: %{customdata[4]}<br>' \
                 '<b>Y_ACCOUNT_CLASS</b>: %{customdata[5]}<br>' \
                 '<b>Y_GL_ACCOUNT_NAME</b>: %{customdata[6]}<br>' \
                 '<b>Y_DMBTR</b>: %{customdata[7]}<br>' \
                 '<b>Y_REC_ERROR</b>: %{customdata[8]}<br>' \
                 '<extra></extra>'

        # add trace to figure
        fig.add_trace(

            # create single scatter plot
            go.Scatter(x=data[z1_col_name], y=data[z2_col_name], mode='markers', marker_color=data[c_col_name], marker=dict(showscale=True, colorbar={'title': '<b>Rec. Error</b>'}), customdata=hover, showlegend=False, hovertemplate=template, opacity=0.7),

            # add scatter plot position
            row=1, col=1

        )

        # set x-axis and y-axis label
        fig['layout']['xaxis{}'.format(str(1))]['title']='<b>[z1]</b>'
        fig['layout']['yaxis{}'.format(str(1))]['title']='<b>[z2]</b>'

        # set colorbar title
        fig['layout']['coloraxis']['colorbar']['title'] = '<b>Rec. Error</b>'

        # iterate over distinct attributes
        for i, attribute in enumerate(attributes):

            # case: continuous attribute
            if attribute == 'Y_DMBTR':

                # prepare color codes
                colors = pd.Categorical(data[attribute]).codes

                # add trace to figure
                fig.add_trace(

                    # create single scatter plot
                    go.Scatter(x=data[z1_col_name], y=data[z2_col_name], mode='markers', marker_color=data[attribute], marker=dict(showscale=False), customdata=hover, showlegend=False, hovertemplate=template, opacity=0.7),

                    # add scatter plot position
                    row=2, col=i+1

                )

            # case: categorical attribute
            else:

                # prepare color codes
                colors = pd.Categorical(data[attribute]).codes

                # add trace to figure
                fig.add_trace(

                    # create single scatter plot
                    go.Scatter(x=data[z1_col_name], y=data[z2_col_name], mode='markers', marker=dict(color=colors, size=8), customdata=hover, showlegend=False, hovertemplate=template, opacity=0.7),

                    # add scatter plot position
                    row=2, col=i+1

                )

            # case: initial plot
            if i == 0:

                # set x-axis and y-axis label
                fig['layout']['xaxis{}'.format(str(i+2))]['title']='<b>[z1]</b>'
                fig['layout']['yaxis{}'.format(str(i+2))]['title']='<b>[z2]</b>'

            # case: non-initial plot
            else:

                # set x-axis label
                fig['layout']['xaxis{}'.format(str(i+2))]['title']='<b>[z1]</b>'

        # save plot to plotting directory
        #fig.write_html(os.path.join(self.plot_dir, filename))

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_interactive(self, parameter, data, hover, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        # cm = plt.cm.get_cmap('RdYlBu')

        # create regular entries scatter plot
        trace1 = px.scatter(data
                            , x=z1_col_name
                            , y=z2_col_name
                            , hover_data=hover
                            , color=c_col_name
                            , color_discrete_map={'regular': 'cornflowerblue', 'global': 'darkorange', 'local': 'red'}
                            , symbol=c_col_name
                            , symbol_sequence=['circle', 'diamond', 'star']
                            )

        # update embedding makers
        trace1.update_traces(marker=dict(size=10, line=dict(width=1, color='white')), selector=dict(mode='markers'))

        # add both scatter plots to figure
        fig = go.Figure(data=trace1.data)

        # update axis layout
        fig.update_layout(
            title=dict(text=title, font_size=28, x=0.5, y=0.95, xanchor='center')
            , margin=dict(l=0, r=0, b=0, t=180, pad=0)
            , xaxis_title='<b>[z1]</b>'
            , xaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            , yaxis_title='<b>[z2]</b>'
            , yaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            , legend_title_text='<b>Classes:</b>'
            , legend={'traceorder': 'reversed'}
        )

        # set colorbar title
        fig['layout']['hoverlabel']['bgcolor'] = 'white'

        # save plot to plotting directory
        fig.write_html(os.path.join(parameter['vis_sub_dir'], filename))

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_error_interactive(self, parameter, data, hover, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        # cm = plt.cm.get_cmap('RdYlBu')

        # determine colorbar start and end range
        range_start = data['Y_REC_ERROR'].mean() - 0.01 * data['Y_REC_ERROR'].std()
        range_end = data['Y_REC_ERROR'].mean() + 0.01 * data['Y_REC_ERROR'].std()

        # create scatter plot
        fig = px.scatter(data
                         , x=z1_col_name
                         , y=z2_col_name
                         , color=c_col_name  # 'Y_ANOMALY_LABEL'
                         , range_color=(range_start, range_end)
                         , color_continuous_scale=px.colors.sequential.Bluered
                         , hover_data=hover
                         , marginal_x='histogram'
                         , marginal_y='histogram')

        # update axis layout
        fig.update_layout(
            title=dict(text=title, font_size=28, x=0.5, y=0.95, xanchor='center')
            , margin=dict(l=0, r=0, b=0, t=180, pad=0)
            , xaxis_title='<b>[z1]</b>'
            , xaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            , yaxis_title='<b>[z2]</b>'
            , yaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
        )

        # set colorbar title
        fig['layout']['coloraxis']['colorbar']['title'] = '<b>Rec. Error</b>'
        fig['layout']['coloraxis']['colorbar']['title']['font']['size'] = 26
        fig['layout']['coloraxis']['colorbar']['tickfont']['size'] = 18
        fig['layout']['hoverlabel']['bgcolor'] = 'white'

        # save plot to plotting directory
        fig.write_html(os.path.join(parameter['vis_sub_dir'], filename))

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_anomalies_interactive(self, parameter, data, hover, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        # cm = plt.cm.get_cmap('RdYlBu')

        data[c_col_name] = data[c_col_name].astype(str)

        # create regular entries scatter plot
        trace1 = px.scatter(data
                            , x=z1_col_name
                            , y=z2_col_name
                            , hover_data=hover
                            , color=c_col_name
                            , color_discrete_map={'1': 'cornflowerblue', '-1': 'red'}
                            , symbol=c_col_name
                            , symbol_sequence=['star', 'circle']
                            )

        # update embedding makers
        trace1.update_traces(marker=dict(size=10, line=dict(width=1, color='white')), selector=dict(mode='markers'))

        # add both scatter plots to figure
        fig = go.Figure(data=trace1.data)

        # update axis layout
        fig.update_layout(
            title=dict(text=title, font_size=28, x=0.5, y=0.95, xanchor='center')
            , margin=dict(l=0, r=0, b=0, t=180, pad=0)
            , xaxis_title='<b>[z1]</b>'
            , xaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            , yaxis_title='<b>[z2]</b>'
            , yaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            , legend_title_text='<b>Classes:</b>'
            , legend={'traceorder': 'reversed'}
        )

        # set colorbar title
        fig['layout']['hoverlabel']['bgcolor'] = 'white'

        # save plot to plotting directory
        fig.write_html(os.path.join(parameter['vis_sub_dir'], filename))

        # close plot
        plt.close()

    # plot the learned embedding in the 2D latent space
    def plot_embeddings_2d_anomalies_cluster_interactive(self, data, hover, z1_col_name, z2_col_name, c_col_name, filename, title):

        # set plotting appearance
        plt.style.use('seaborn')
        # plt.rcParams['figure.figsize'] = [9, 11]  # width * height
        plt.rcParams['figure.figsize'] = [6, 6]  # width * height
        plt.rcParams['agg.path.chunksize'] = 1000000
        # cm = plt.cm.get_cmap('RdYlBu')

        # add random jitter to z1 and z2 coordinates
        #data['z1'] = self.rand_jitter(data['z1'])
        #data['z2'] = self.rand_jitter(data['z2'])

        # data['z1'] = data['z1'] + np.random.randn() * 0.01
        # data['z2'] = data['z2'] + np.random.randn() * 0.01

        # extract the detected cluster
        cluster = data[data['Y_ANOMALY_CLASS'] != -1]

        # extract the detected anomalies
        anomalies = data[data['Y_ANOMALY_CLASS'] == -1]

        # convert the detected cluster into categorical
        cluster['Y_ANOMALY_CLASS'] = cluster['Y_ANOMALY_CLASS'].astype('category')

        # scatter plot of embeddings
        trace1 = px.scatter(cluster
                            , x=z1_col_name
                            # , marginal_x='histogram'
                            , y=z2_col_name
                            # , marginal_y='histogram'
                            , color=c_col_name
                            , hover_data=hover)

        # update embedding makers
        trace1.update_traces(marker=dict(size=10, line=dict(width=1, color='gray')), selector=dict(mode='markers'))

        # scatter plot of anomalies
        trace2 = px.scatter(anomalies
                            , x=z1_col_name
                            , y=z2_col_name
                            , hover_data=hover)

        # update anomaly markers
        trace2.update_traces(marker=dict(size=10, symbol='diamond', line=dict(width=5, color='red')), selector=dict(mode='markers'))

        # add both scatter plots to figure
        fig = go.Figure(data=trace1.data + trace2.data)

        # update axis layout
        fig.update_layout(
            title=dict(text=title, font_size=28, x=0.5, xanchor='center')
            , hoverlabel=dict(bgcolor="white", font_size=14)
            , xaxis_title='<b>[z1]</b>'
            , xaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            , yaxis_title='<b>[z2]</b>'
            , yaxis=dict(titlefont=dict(size=26), tickfont=dict(size=18))
            , legend_title_text='<b>Cluster</b>'
            , legend={'traceorder': 'reversed'}
        )

        # save plot to plotting directory
        #fig.write_html(os.path.join(self.plot_dir, filename))

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

    def rand_jitter(self, attribute, jitter=.015):

        # set random seed
        np.random.seed(1234)

        # determine attribute std
        std = attribute.std()

        # add random jitter to attribute
        attribute = attribute + np.random.normal(0.0, std, len(attribute)) * jitter

        # return jittered input
        return attribute
