import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
from dash import Dash, dcc, html

train_df = pd.read_csv('f_train.csv')

print(
    "Observation/Conclusion:\nWe have got the almost same number of readings\nAs showed in the figure, it can be clearly found that the data is more balanced, indicating that the data set can be better suitable for building any Data Science/Machine Learning project.")

fig1 = px.histogram(train_df, x="Activity", color="Activity", title="Number of data points in each activity")
# fig1.show()


print(
    "Observation/Conclusion:\nMoreover, it can be concluded from this plot that the error range of the six activities is between 2% and 5%.")
fig2 = px.histogram(train_df, x="subject", color="Activity", barmode='group',
                    title="Data provided by each user/subject")
# fig.show()


print("Motionless and Dynamic Activities\
\n1) In motionless activities (sit, stand, lie down) motion\
 information will not be very useful.\
\n2) In the dynamic activities (Walking, WalkingUpstairs,\
WalkingDownstairs) motion info will be sign")

print("Stationary and Moving activities are completely different\
\nBased on the given data, the graphs of six active states are\
 drawn, as shown in the distrubution plot. We can find that motionless\
 activities are more intensive, while dynamic activities are\
 relatively less. This reason can explain to a great extent that\
 in today’s life, people may exercise very little because of\
 the pressure of life, thus affecting their physical and mental health.\n\
")

body_acc_mag_mean = train_df[['tBodyAccMagmean', 'Activity']]
group_labels = body_acc_mag_mean['Activity'].unique().tolist()
hist_data = []
for i in group_labels:
    hist_data.append(body_acc_mag_mean[body_acc_mag_mean['Activity'] == i]['tBodyAccMagmean'].values)
# Create distplot with curve_type set to 'normal'
fig3 = ff.create_distplot(hist_data, group_labels, show_hist=False)

# Add title
fig3.update_layout(title_text='Distribution plot of all activities using Average Body Acceleration values',
                   yaxis_title="Probability Density", xaxis_title="tBodyAccMagmean")
# fig3.show()


body_acc_mag_mean = train_df[['tBodyAccMagmean', 'Activity']]
stationary_labels = body_acc_mag_mean['Activity'].unique().tolist()[0:3]  # sitting standing and laying
hist_data_stationary = []
for i in stationary_labels:
    hist_data_stationary.append(body_acc_mag_mean[body_acc_mag_mean['Activity'] == i]['tBodyAccMagmean'].values)
# Create distplot with curve_type set to 'normal'
fig4 = ff.create_distplot(hist_data_stationary, stationary_labels, show_hist=False)

# Add title
fig4.update_layout(
    title_text='Distribution plot of Stationary activities using Average Body Acceleration values  (Laying,Sitting and Standing)',
    yaxis_title="Probability Density", xaxis_title="tBodyAccMagmean")
# fig4.show()


body_acc_mag_mean = train_df[['tBodyAccMagmean', 'Activity']]
moving_labels = body_acc_mag_mean['Activity'].unique().tolist()[3:6]  # walking,walking downstairs,walking upstairs
hist_data_moving = []
for i in moving_labels:
    hist_data_moving.append(body_acc_mag_mean[body_acc_mag_mean['Activity'] == i]['tBodyAccMagmean'].values)
# Create distplot with curve_type set to 'normal'
fig5 = ff.create_distplot(hist_data_moving, moving_labels, show_hist=False)

# Add title
fig5.update_layout(
    title_text='Distribution plot of moving activities on the Body Acceleration Mean column (Walking,Walking Down and Walking Up)',
    yaxis_title="Probability Density", xaxis_title="tBodyAccMagmean")
# fig5.show()

print("1) If tAccMean is < -0.8 then the Activities are either Standing or Sitting or Laying.\
\n2) If tAccMean is > -0.6 then the Activities are either Walking or WalkingDownstairs or Walking Upstairs.\
\n3) If tAccMean > 0.0 then the Activity is Walking Downstairs.\
\n4) We can classify 75% the Acitivity labels with some errors")

fig6 = px.box(train_df, x='Activity', y='tBodyAccMagmean', color='Activity')
fig6.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
# fig6.show()


print("The maximum acceleration is a better variable to distinguish non-moving activities (blue 'standing', red 'sitting' and green 'laying') from moving activities (purple 'walk', prange 'walkdown' and lightblue 'walkup'\
EXPECTATIONS: Clustering analysis on maximum acceleration should distinguish better the activities")

sub1 = train_df[train_df['subject'] == 1]
fig7 = px.scatter(sub1, x=sub1.index, y=sub1.columns[9], color='Activity')
# fig7.show()
fig8 = px.scatter(sub1, x=sub1.index, y=sub1.columns[10], color='Activity')
# fig7.show()
fig9 = px.scatter(sub1, x=sub1.index, y=sub1.columns[11], color='Activity')
# fig9.show()


print(
    'Calculate the distance between the average acceleration in the x, y, and z-coordinates and all of the different activities, and perform a hierarchical clustering')
print(
    'CONCLUSIONS:\n Clustering analysis does not  distinguish the activities well. For non-moving activities, only walkdown activities (in lightblue) have his own cluster')
sub1 = train_df[train_df['subject'] == 1][['tBodyAccmaxX', 'tBodyAccmaxY', 'tBodyAccmaxZ', 'subject', 'Activity']]
d = squareform(pdist(sub1.iloc[:, 0:3].values, 'euclidean'))
fig10 = ff.create_dendrogram(d, color_threshold=1.5)
fig10.update_layout(title_text="Cluster dendrogram (max acceleration)", width=800, height=500,
                    xaxis_title="distance matrix", yaxis_title="height")
# fig10.show()


app = Dash(__name__)


server = app.server
app.layout = html.Div(children=[
    html.H1(children='Data Visualization of Human Activity Recognition Dataset'),

    html.Div(children='''
        The following dataset was recorded using an accelerometer and other sensors in a smartphone. The columns are various metrics used to describe each activity - Standing,Sitting,Laying,Walking,Walking downstairs,Walking upstairs.
    '''),

    html.Div(children='''
       In a smart urban environment, providing accurate information about human activities is an
important task. It is a trend to implement human activity recognition (HAR) algorithms and applications on
smart phones, including health monitoring, self-management system, health tracking and so on. 
    '''),

    html.Div(children='''
       Let's explore and analyse the dataset. 
    '''),

    dcc.Graph(
        id='fig1',
        figure=fig1

    ),

    html.Div(children='''
        Conclusion:\nWe have got the almost same number of readings. It's balanced.\nAs showed in the figure, it can be clearly found that the data is more balanced, indicating that the data set can be better suitable for building any Data Science/Machine Learning project.
    '''),

    dcc.Graph(
        id='fig2',
        figure=fig2
    ),

    html.Div(children='''
         Conclusion:\nMoreover, it can be concluded from this plot that the error range of the six activities is between 2% and 5%.
    '''),

    dcc.Graph(
        id='fig3',
        figure=fig3
    ),

    dcc.Graph(
        id='fig4',
        figure=fig4
    ),
    dcc.Graph(
        id='fig5',
        figure=fig5
    ),

    html.Div(children='''
         Conclusion:\nMotionless and Dynamic Activities\
\n1) In motionless activities (sit, stand, lie down) motion\
 information will not be very useful.\
\n\n2) In the dynamic activities (Walking, WalkingUpstairs,\
WalkingDownstairs) motion info will be sign.\n
\nStationary and Moving activities are completely different\
\nBased on the given data, the graphs of six active states are\
 drawn, as shown in the distrubution plot. We can find that motionless\
 activities are more intensive, while dynamic activities are\
 relatively less. This reason can explain to a great extent that\
 in today’s life, people may exercise very little because of\
 the pressure of life, thus affecting their physical and mental health.\n\
    '''),

    dcc.Graph(
        id='fig6',
        figure=fig6
    ),

    html.Div(children='''
    1) If tAccMean is < -0.8 then the Activities are either Standing or Sitting or Laying.\n\
\n2) If tAccMean is > -0.6 then the Activities are either Walking or WalkingDownstairs or Walking Upstairs.\n\
\n3) If tAccMean > 0.0 then the Activity is Walking Downstairs.\n\
\n4) We can classify 75% the Acitivity labels with some errors
    '''),

    dcc.Graph(
        id='fig7',
        figure=fig7
    ),

    dcc.Graph(
        id='fig8',
        figure=fig8
    ),

    dcc.Graph(
        id='fig9',
        figure=fig9
    ),

    html.Div(children='''
        Conclusion:\nThe maximum acceleration is a better variable to distinguish non-moving activities (blue 'standing', red 'sitting' and green 'laying') from moving activities (purple 'walk', prange 'walkdown' and lightblue 'walkup'\
EXPECTATIONS: Clustering analysis on maximum acceleration should distinguish better the activities
    '''),

    dcc.Graph(
        id='fig10',
        figure=fig10
    ),

    html.Div(children='''
        Calculate the distance between the average acceleration in the x, y, and z-coordinates and all of the different activities, and perform a hierarchical clustering\n
      CONCLUSIONS:\n Clustering analysis does not  distinguish the activities well. For non-moving activities, only walkdown activities have his own cluster\n 

    ''')

])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)