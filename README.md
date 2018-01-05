# Gym Renju DQN

Sample implementation of Renju AI for gym-renju.
The AI is implemented with DQN algorithm on Chainer.

This is just an sample implementation of AI, so you can create more smart AI with modifying some parameters or implementaions.

Please also refer to gym-renju.

- <https://github.com/aratakokubun/gym_renju>
- <https://pypi.python.org/pypi/gym-renju/>

## Dependencies

- gym-renju(=0.1.8)
- Chainer(>=1.15, <https://chainer.org/>)

## Run

### Run with learned model

```python
python src/run.py
```

### Train

```python
python src/run.py --train
```

## Reference

This implementation are basically based on chainer_pong by icoxfog417.
Thanks!

- <https://github.com/icoxfog417/chainer_pong>