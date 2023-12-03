# # import pygame

# # class Simulation:
# #     def __init__(self):
# #         self.width = 1360
# #         self.height = 760
# #         self.title = "Simulation"

# #         self.screen_size = (self.width,self.height)
# #         self.center = (self.width/2,self.height/2)

# #         pygame.init()
# #         pygame.display.set_caption(self.title)
# #         self.font = pygame.font.SysFont("arial", 32)
# #         self.text = self.font.render('Simulation', True, (255,255,255), (96,96,96))
# #         self.text_rec = self.text.get_rect()
# #         self.text_rec.center = (680,40)
# #         self.screen = pygame.display.set_mode(self.screen_size)
# #         self.clock = pygame.time.Clock()
    
# #     def run(self):
# #         self.screen.fill( (96,96,96) )
# #         self.screen.blit(self.text,self.text_rec)
# #         # (192,192,192)
# #         pygame.draw.rect(self.screen, (255,255,255) , pygame.Rect( (80,80), (1200,600)))
# #         start_time = pygame.time.get_ticks()
# #         while True:
# #             for event in pygame.event.get():
# #                 if event.type == pygame.QUIT:
# #                     pygame.quit()
# #                     return
# #             pygame.display.update()
# #             self.clock.tick(60)

# # from src.ClusterMember import ClusterMember
# from src.utils import *
# from src.ClusterAlgo import Kmeans
# # from src.Simulation import Simulation
# from matplotlib import gridspec
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import numpy as np

# # cm = ClusterMember((1.0,2.0,3.0), Mobility("stationary"))
# # print(cm.printPosition())
# # print(cm.printMobility())

# d = Distribution()
# data = d.getDistribution('uniform', [(0,0), (100,100), (50,2)])
# # model = Kmeans(data).generateModel(optimal=True)
# # print(model.cluster_centers_)

# # s = Simulation()
# # s.run()

# fig = plt.figure(figsize=(14, 7))
# gs = gridspec.GridSpec(2, 2, width_ratios=[7, 3])
# ax = plt.subplot(gs[:,0])
# ax2 = plt.subplot(gs[0,1])
# ax3 = plt.subplot(gs[1,1])
# # fig, ax = plt.subplots(1,1)

# def animate(time):
#     print(time)
#     if time%10 == 0:
#         ax.cla()
#         data = d.getDistribution("randint", [(0,0), (100,100), (100,2)])
#         x,y = data.T
#         ax.scatter(x,y)
#         ax.set_xlim([0,100])
#         ax.set_ylim([0,100])
#     if time%10 == 3:
#         ax2.cla()
#         data = d.getDistribution("randint", [(0,0), (100,100), (100,2)])
#         x,y = data.T
#         ax2.scatter(x,y)
#         ax2.set_xlim([0,100])
#         ax2.set_ylim([0,100])
    
#     if time%10 == 6:
#         ax3.cla()
#         data = d.getDistribution("randint", [(0,0), (100,100), (100,2)])
#         x,y = data.T
#         ax3.scatter(x,y)
#         ax3.set_xlim([0,100])
#         ax3.set_ylim([0,100])

# def init_ani():
#     ax.cla()
#     ax2.cla()
#     ax3.cla()
#     data = d.getDistribution("randint", [(0,0), (100,100), (100,2)])
#     x,y = data.T
#     ax.scatter(x,y)
#     ax.set_xlim([0,100])
#     ax.set_ylim([0,100])
#     ax2.scatter(x,y)
#     ax2.set_xlim([0,100])
#     ax2.set_ylim([0,100])
#     ax3.scatter(x,y)
#     ax3.set_xlim([0,100])
#     ax3.set_ylim([0,100])


# ani = FuncAnimation(fig, animate, frames=60, init_func=init_ani, interval=500)
# plt.show()

