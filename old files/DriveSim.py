#########################################################
#
#   DriveSim.py
#   © 2022.9 Yeonjun Kim (김연준) <kyjunclsrm@gmail.com>
#
#########################################################

from asynchat import simple_producer
from cmath import pi
import collections
from email.errors import InvalidMultipartContentTransferEncodingDefect
import numpy as np
import pygame
import random
import os
import time
import math

import matplotlib.pyplot as plt

from Network_PER import *
        

class DriveSimulator(object):
    def __init__(self):
        # Initialize pygame
        #os.environ["DSL_VIDEODRIVER"] = "dummy"
        pygame.init()

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
        self.STATUS_H = 100 # Agent status monitor below screen

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
        self.agtRot = -math.pi/2
        self.agtV = 0.0
        self.agtRect = pygame.Rect(self.agtPos, self.agtSize)
        self.agtImg_org = pygame.image.load("carimg.png").convert()
        self.agtImg = pygame.transform.rotate(self.agtImg_org, self.agtRot*180/math.pi)
        self.agtRwd = 0.0 # 축적된 보상
        
        # Obs(Obstacle)
        self.obsRad = random.randint(75,125)
        self.obsPos = (600, random.randint(self.obsRad, self.SCREEN_H - self.obsRad))

        # Initialize sim_state
        self.sim_state = np.array([])
        self.sim_state = self.get_sim_state() #가장 처음 상태 구하기

    def get_obs_dir(self):
        # Get the direction to obstacle (in radian)
        dx = (self.agtPos[0] + self.agtSize[0]/2) - self.obsPos[0]
        dy = (self.agtPos[1] + self.agtSize[1]/2) - self.obsPos[1]
        if dy == 0.0:
            dy = 0.01 # Prevents Division by Zero Error
        theta = math.atan(dx/dy)

        # Adjust theta (range of atan(x) = (-pi/2, pi/2))
        if dy < 0:
            # if agent is located higher than obstacle
            if theta < 0:
                theta += math.pi
            else:
                theta -= math.pi

        return theta - self.agtRot

    def get_obs_dist(self):
        # Get the distance between agent and obstacle
        ax = self.agtPos[0] + self.agtSize[0]/2
        ay = self.agtPos[1] + self.agtSize[1]/2
        dsquare = (ax - self.obsPos[0])**2 + (ay - self.obsPos[1])**2
        return math.sqrt(dsquare) - self.obsRad - 20

    def get_sim_state(self):
        sim_state = self.sim_state

        sim_cur_state = np.array([
            self.agtV, #에이전트 속력
            (self.agtPos[0]-self.FINISH_LANE)/200.0, # 도착점까지 거리
            (self.agtPos[1]-self.CENTER_LANE)/200.0, # 중앙 차선까지 거리
            self.agtRot, #에이전트 방향 (도착점 기준)
            self.get_obs_dist()/200.0, #장애물까지 거리
            self.get_obs_dir()]) #에이전트 방향(장애물 기준)

        #if sim_state.size == 0:
        #    sim_state = np.array([sim_cur_state, sim_cur_state, sim_cur_state, sim_cur_state])
        #else:
        #    sim_state = np.array([sim_cur_state, sim_state[0], sim_state[1], sim_state[2]])

        sim_state = sim_cur_state
        return sim_state

    def step(self, action):
        pygame.event.pump()
        self.t += 1

        # (1)
        # 에이전트 행동 반영 (action = 0(그대로), 1(엑셀), 2(브레이크), 3(좌회전), 4(우회전))
        if action == 0:
            pass
        elif action == 1:
            self.agtV += 0.5
        elif action == 2:
            self.agtV -= 0.5
        elif action == 3:
            self.agtRot += math.pi/24
        elif action == 4:
            self.agtRot -= math.pi/24
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
        # 에이전트 그리기
        x, y = self.agtPos
        x -= self.agtV * math.sin(self.agtRot)
        y -= self.agtV * math.cos(self.agtRot)
        self.agtPos = (x, y)

        self.agtImg = pygame.transform.rotate(self.agtImg_org, self.agtRot*180/math.pi)
        self.screen.blit(self.agtImg, self.agtPos)
        # 장애물 그리기
        pygame.draw.circle(self.screen, self.COLOR_OBS, self.obsPos, self.obsRad)
        

        # (4)
        # 보상 결정하기
        self.stpRwd = 0.0 # Step Reward
        self.sim_over = False
        self.sim_over_why = ''

        # 장애물을 성공적으로 회피한 경우
        if self.agtPos[0] > self.FINISH_LANE:
            self.sim_over = True
            self.sim_over_why = '장애물 회피 성공'
            self.win_count += 1
            self.stpRwd = 5.0

        # 회피하지 못한 경우 3가지
        if self.get_obs_dist() < 0:
            self.sim_over = True
            self.sim_over_why = '장애물과 충돌'
            self.stpRwd = -3.0
        
        if self.agtPos[1] < 0 or self.agtPos[1] + self.agtSize[1] > self.SCREEN_H or self.agtPos[0] < 0:
            self.sim_over = True
            self.sim_over_why = '경로 이탈'
            self.stpRwd = -3.0
            
        if self.t >= 400: #300 Ticks 안에 목표에 도달하지 못하면 종료
            self.sim_over = True
            self.sim_over_why = '시간 초과'
            self.stpRwd = -3.0

        if self.sim_over:
            #print(abs(self.agtPos[1]-self.CENTER_LANE)/self.CENTER_LANE)
            #print(2*(self.agtPos[0]-self.FINISH_LANE)/self.FINISH_LANE)
            #print('------')
            # 중앙선으로부터 떨어진 정도에 따라 음의 보상
            self.stpRwd -= abs(self.agtPos[1]-self.CENTER_LANE)/(self.CENTER_LANE*2)
            # 목표지점과의 거리 기준으로 음의 보상
            self.stpRwd += 2*(self.agtPos[0]-self.FINISH_LANE)/self.FINISH_LANE

        self.agtRwd += self.stpRwd # 누적 보상 저장


        #Update sim_state
        self.sim_prev_state = self.sim_state
        self.sim_state = self.get_sim_state()
        
        pygame.display.flip()
        self.clock.tick(self.frame_rate)

        if(self.sim_over or self.t % 3 == 0): #Frame Skipping (3프레임마다 의사 결정)
            return self.sim_state, self.stpRwd, self.sim_over
        else:
            self.step(0)
            return self.sim_state, self.stpRwd, self.sim_over
    
    def quit(self):
        pygame.quit()
