import math
import numpy as np
import warnings
from torch.optim.lr_scheduler import _LRScheduler


class CustomCosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CustomCosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


import math
import numpy as np
import warnings


class TeacherForcingScheduler:
    """Teacher Forcing 스케줄러 클래스. Train에 활용
    Example:
        # Define TF Scheduler
        total_steps = len(train_data_loader)*options.num_epochs
        teacher_forcing_ratio = 0.6
        tf_scheduler = TeacherForcingScheduler(
            num_steps=total_steps,
            tf_max=teacher_forcing_ratio
            tf_min=0.4
            )

        # Train phase
        tf_ratio = tf_scheduler.step()
        output = model(input, expected, False, tf_ratio)

    Args:
        num_steps (int): 총 스텝 수
        tf_max (float): 최대 teacher forcing ratio. tf_max에서 시작해서 코사인 함수를 그리며 0으로 마무리 됨
        tf_min (float, optional): 최소 teacher forcing ratio. Defaults to 0.4
    """

    def __init__(self, num_steps: int, tf_max: float = 1.0, tf_min: float = 0.4):
        linspace = self._get_arctan(num_steps, tf_max, tf_min)
        self.__scheduler = iter(linspace)
        self.tf_max = tf_max
        self.tf_min = tf_min

    def step(self):
        try:
            return next(self.__scheduler)
        except:
            # 스케줄링이 끝났는데 학습은 종료되지 않은 경우 tf_min을 리턴
            warnings.warn(
                f"Teacher forcing scheduler has been done. Return just tf_min({self.tf_min}) for now."
            )
            return self.tf_min

    @staticmethod
    def _get_arctan(num_steps: int, tf_max: float, tf_min: float):
        diff = tf_max - tf_min
        inflection = int(num_steps * 0.1)
        x = np.linspace(-5, 5, num_steps)  # NOTE. for transformer
        x = -np.arctan(x)
        x -= x[-1]
        x *= diff / x[0]
        x += tf_min
        x = x[inflection:]
        return x

    @staticmethod
    def _get_cosine(num_steps: int, tf_max: float):  # NOTE. 아직 tf_min 미적용. 무조건 0으로 하강함
        factor = tf_max / 2
        x = np.linspace(0, np.pi, num_steps)
        x = np.cos(x)
        x *= factor
        x += factor
        return x
