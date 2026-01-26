# Main Model (Model 2)

## Purpose
This directory contains the main model used in the thesis.
The purpose of this model is to analyze under what conditions
learning-like improvement can emerge from the temporal structure
of electrical current responses observed in an EAP gel.

Rather than assuming learning, this model explicitly defines
current dynamics and evaluates how performance changes in a Pong task
can be interpreted as learning under specific structural conditions.

## Model Description
- The gel is divided into three spatial regions
- Each region produces a normalized electrical current response
- Paddle position is determined by fitting a quadratic function
  to the three current values
- Learning-like behavior is evaluated through the evolution of
  current differences between regions

Both independent and dependent current behaviors are examined.

## Files
- `main_model.py` : main simulation script

## How to Run
```bash
python main_model.py
