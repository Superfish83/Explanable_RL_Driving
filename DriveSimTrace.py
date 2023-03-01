#########################################################
#
#   DriveSim.py
#   © 2022.9 Yeonjun Kim (김연준) <kyjunclsrm@gmail.com>
#
#########################################################

from asynchat import simple_producer
import collections
from email.errors import InvalidMultipartContentTransferEncodingDefect
import numpy as np
import pygame
import random
import os
import time
from math import *

import matplotlib.pyplot as plt

from Network_PER import *
        

class DriveSimulator(object):
    def __init__(self):
        # Initialize pygame
        #os.environ["DSL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()

        # Initialize Constants . . .
        # Colors
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_AGENT = (102,255,255)
        self.COLOR_PATH = (255,255,102)
        self.COLOR_OBS = (204,0,0)

        # Screen size
        self.SCREEN_W = 1000
        self.SCREEN_H = 500
        self.STATUS_H = 300 # Agent status monitor below screen

        # Traffic lanes
        self.CENTER_LANE = self.SCREEN_H/2
        self.FINISH_LANE = self.SCREEN_W-100

        # Initialize episode count
        self.episode_count = 0
        self.win_count = 0


    def reset(self, frame_rate):
        self.episode_count += 1
        self.frame_rate = frame_rate

        # Initialize variables
        self.state = np.array([])
        self.prev_state = np.array([])
        self.t = 0
        self.isOver = False
        self.whyOver = ''

        # Initialize...

        # Screen
        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H + self.STATUS_H))
        self.clock = pygame.time.Clock()

        # Agent
        self.agtSize = (60,40)
        self.agtPos = (100, self.CENTER_LANE - self.agtSize[1]/2)
        self.agtRot = -pi/2
        self.agtV = 4.0
        self.agtRect = pygame.Rect(self.agtPos, self.agtSize)
        self.agtImg_org = pygame.image.load("carimg.png").convert()
        self.agtImg = pygame.transform.rotate(self.agtImg_org, self.agtRot*180/pi)
        self.agtRwd = np.zeros(3) # 축적된 보상
        self.sensing_range = 50 # 상태 관측 시 장애물 감지 범위
        
        # Obs(Obstacle)
        self.obsRad = random.randint(100,150)
        self.obsPos = (600 + random.randint(-50,50), random.randint(self.obsRad, self.SCREEN_H - self.obsRad))

        # Initialize sim_state
        self.sim_state = np.array([])
        self.sim_state = self.get_sim_state() #가장 처음 상태 구하기

        self.trace = []
        self.explain = False #Explain Mode 여부

    # def get_obs_dir(self):
    #     # Get the direction to obstacle (in radian)
    #     dx = (self.agtPos[0] + self.agtSize[0]/2) - self.obsPos[0]
    #     dy = (self.agtPos[1] + self.agtSize[1]/2) - self.obsPos[1]
    #     if dy == 0.0:
    #         dy = 0.01 # Prevents Division by Zero Error
    #     theta = atan(dx/dy)

    #     # Adjust theta (range of atan(x) = (-pi/2, pi/2))
    #     if dy < 0:
    #         # if agent is located higher than obstacle
    #         if theta < 0:
    #             theta += pi
    #         else:
    #             theta -= pi

    #     return theta - self.agtRot

    def get_obs_dist(self):
         # Get the distance between agent and obstacle
         ax = self.agtPos[0] + self.agtSize[0]/2
         ay = self.agtPos[1] + self.agtSize[1]/2
         dsquare = (ax - self.obsPos[0])**2 + (ay - self.obsPos[1])**2
         return sqrt(dsquare) - self.obsRad - 20


    # def get_obs_sense(self):
    #     r = self.sensing_range
    #     theta = self.agtRot

    #     sense = [0,0,0,0]
    #     sense_index = [0,1,1,1,2,2,2,2,3,3,3,0]
    #     #감지된 정보가 sense의 어느 index에 저장될지 결정
    #     #0: 정면, 1/3: 측면, 2: 후면

    #     a_x, a_y = (self.agtPos[0] + self.agtSize[0]/2), (self.agtPos[1] + self.agtSize[1]/2)
    #     o_x, o_y, o_r = self.obsPos[0], self.obsPos[1], self.obsRad
    #     for i in range(12):
    #         x = a_x + r*cos(theta + pi*(i/6))
    #         y = a_y + r*sin(theta + pi*(i/6))
    #         if (x-o_x)**2 + (y-o_y)**2 < o_r ** 2:
    #             sense[sense_index[i]] = 1
        
    #     return sense

    def get_sim_state(self):
        sim_cur_state = np.array([
            (self.agtV * 200), # 에이전트 속력
            ((self.agtRot + pi/2) * 200), #에이전트가 바라보는 방향
            (self.agtPos[0]-self.FINISH_LANE), # 에이전트 위치 (x)
            (self.agtPos[1]-self.CENTER_LANE), # 에이전트 위치 (y)
            (self.obsPos[0] - self.agtPos[0]), # 장애물 위치 (x)
            (self.obsPos[1] - self.agtPos[1]), # 장애물 위치 (y)
            (self.obsRad)])                    # 장애물 크기
        # sense = self.get_obs_sense()
        # sim_cur_state = np.array([
        #     (self.agtV), # 에이전트 속력
        #     (self.agtPos[0]-self.FINISH_LANE), # 에이전트 위치 (x)
        #     (self.agtPos[1]-self.CENTER_LANE), # 에이전트 위치 (y)
        #     (sense[0]),
        #     (sense[1]),
        #     (sense[2]),
        #     (sense[3])])                    
        # print(sense)

        sim_cur_state /= 200.0 #정규화(normalization)

        self.sim_state = np.array([sim_cur_state])

        #sim_state = self.sim_state
        #if sim_state.size == 0:
        #    self.sim_state = np.array([sim_cur_state, sim_cur_state])
        #else:
        #    self.sim_state = np.array([sim_cur_state, sim_state[0]])

        return self.sim_state





        

    def render_explanation(self, shap_list, pred_list, shap_exp, pred_exp):
        font1 = pygame.font.SysFont('NanumGothic', 20)
        font2 = pygame.font.SysFont('NanumGothicBold', 20)
        
        text_surface = font1.render(pred_exp[0], False, (255,255,255))
        self.screen.blit(text_surface, (20, self.SCREEN_H + 25))

        text_surface = font2.render(pred_exp[1], False, (255,255,255))
        self.screen.blit(text_surface, (20, self.SCREEN_H + 50))

        text_surface = font2.render(shap_exp[0], False, (255,255,255))
        self.screen.blit(text_surface, (20, self.SCREEN_H + 75))

        text_surface = font1.render(shap_exp[1], False, (255,255,255))
        self.screen.blit(text_surface, (20, self.SCREEN_H + 100))



        pygame.display.flip()



    def step(self, action, pred_C):
        pygame.event.pump()
        self.t += 1

        # (1)
        # 에이전트 행동 반영 (action = 0(그대로), 1(엑셀), 2(브레이크), 3(좌회전), 4(우회전))
        if action == 0:
            pass
        elif action == 1:
            self.agtV += 0.5
        elif action == 2:
            self.agtV *= 0.9
        elif action == 3:
            self.agtRot += pi/24
        elif action == 4:
            self.agtRot -= pi/24
        #에이전트 업데이트


        
        # (2)
        # 화면 그리기
        self.screen.fill(self.COLOR_BLACK)
        # 각종 표시선 그리기
        pygame.draw.line(self.screen, self.COLOR_WHITE,
                [self.FINISH_LANE,0], [self.FINISH_LANE, self.SCREEN_H], 4)
        pygame.draw.line(self.screen, self.COLOR_PATH,
                [0,self.CENTER_LANE], [self.SCREEN_W, self.CENTER_LANE], 4)
        pygame.draw.rect(self.screen, self.COLOR_PATH,
                [(100, self.CENTER_LANE - self.agtSize[1]/2),self.agtSize], 4)
        pygame.draw.line(self.screen, self.COLOR_WHITE,
                [0,self.SCREEN_H], [self.SCREEN_W,self.SCREEN_H])


        # 에이전트 경로 그리기
        cnt = 0
        for pos in self.trace:
            cnt+=1
            pos = (
                pos[0] + self.agtSize[0] / 2,
                pos[1] + self.agtSize[1] / 2
            )
            #c = max(0, 255 - 5 * (len(self.trace) - cnt))
            #c = min(255, 30 * (len(self.trace) - cnt))
            if self.explain:
                pygame.draw.circle(self.screen, (150,150,250), pos, 5)
            else:
                pygame.draw.circle(self.screen, (220,220,220), pos, 5)


        # 에이전트 그리기
        x, y = self.agtPos
        x -= self.agtV * sin(self.agtRot)
        y -= self.agtV * cos(self.agtRot)
        self.agtPos = (x, y)

        if self.t % 3 == 0:
            self.trace.append(self.agtPos)

        self.agtImg = pygame.transform.rotate(self.agtImg_org, self.agtRot*180/pi)
        self.screen.blit(self.agtImg, self.agtPos)
        # 장애물 그리기
        pygame.draw.circle(self.screen, self.COLOR_OBS, self.obsPos, self.obsRad)





        # (4)
        # 보상 결정하기
        self.stpRwd = np.zeros(3) # Step Reward [목표도착, 장애물충돌, 경로이탈, 시간초과, 과속] -> Reward 분리
        self.sim_over = False
        self.sim_over_why = ''

        # 장애물을 성공적으로 회피한 경우
        if self.agtPos[0] > self.FINISH_LANE:
            self.sim_over = True
            self.sim_over_why = '장애물 회피 성공'
            self.win_count += 1
            self.stpRwd[0] = 1.0

        # 회피하지 못한 경우
        if self.get_obs_dist() < 0:
            self.sim_over = True
            self.sim_over_why = '장애물과 충돌'
            self.stpRwd[1] = -1.0
        
        if self.agtPos[1] < 0 or self.agtPos[1] + self.agtSize[1] > self.SCREEN_H or self.agtPos[0] < 0:
            self.sim_over = True
            self.sim_over_why = '경로 이탈'
            self.stpRwd[2] = -1.0
            
        if self.t >= 400: #400 Ticks 안에 목표에 도달하지 못하면 종료
            self.sim_over = True
            self.sim_over_why = '시간 초과'
            self.stpRwd[0] = (self.agtPos[0] - self.obsPos[0]) / 500 # 목적지로부터 가까운 만큼 보상

        #if self.agtV <= 1.0: #움직이지 않을 시
        #    self.stpRwd[3] = -0.03

        #if self.agtV >= 7.0: #과속 시
        #    self.stpRwd[3] = -0.03
        #    text_surface = my_font.render("High Speed!", False, (255,255,255))
        #    self.screen.blit(text_surface, (self.agtPos[0]+60, self.agtPos[1]+30))




        # 누적된 보상 표시
        # +) 에이전트 판단 근거(Decomposed Reward 예측값) 표시

        my_font = pygame.font.SysFont('NanumGothic', 20)
        if not self.explain:
            self.agtRwd += self.stpRwd # 누적 보상 저장
            text_surface = my_font.render(f"Accumulated Reward: {self.agtRwd}", False, (255,255,255))
            self.screen.blit(text_surface, (20, self.SCREEN_H + 20 + 25*len(pred_C)))

            decomposition = ['Reach Finish Line', 'Collide with Obstacle', 'Collide with Wall']
            if len(pred_C) > 0:
                for i in range(len(pred_C)):
                    pred = np.array(pred_C[i])[0]
                    
                    txt = decomposition[i] + ' :'
                    text_surface = my_font.render(txt, False, (255,255,255))
                    self.screen.blit(text_surface, (20, self.SCREEN_H + 20+ 25*i))
                    for j in range(5):
                        txt = str(round(pred[j],2))
                        text_surface = my_font.render(txt, False, (255,255,255))
                        self.screen.blit(text_surface, (260+70*j, self.SCREEN_H + 20+ 25*i))

        
        pygame.display.flip()
        self.clock.tick(self.frame_rate)

        if(self.sim_over or self.t % 3 == 0): #Frame Skipping (3프레임마다 의사 결정)
            #Update sim_state
            self.sim_prev_state = self.sim_state
            self.sim_state = self.get_sim_state()
            return self.sim_state, self.stpRwd, self.sim_over
        else:
            r = self.stpRwd
            s, r_, o = self.step(0, pred_C)
            return s, r+r_, o
    
    def quit(self):
        pygame.quit()
