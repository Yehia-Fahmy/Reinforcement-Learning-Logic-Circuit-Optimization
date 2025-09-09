#!/bin/zsh

python random_agent.py --base arithmetic --name adder
python random_agent.py --base arithmetic --name bar
python random_agent.py --base arithmetic --name sin
python random_agent.py --base random_control --name i2c
python random_agent.py --base random_control --name int2float
python random_agent.py --base random_control --name router
python random_agent.py --base random_control --name ctrl
python random_agent.py --base random_control --name dec
python random_agent.py --base random_control --name priority
python random_agent.py --base random_control --name cavlc 