#!/usr/bin/env python
# coding: utf-8

# In[4]:


import flwr as fl

from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional
from typing import Tuple, Union, List

import os
os.system("python C:/Users/siddh/")
import utils


# In[5]:


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""
    _, (X_test, y_test) = utils.load_cleveland()

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# In[ ]:


if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,)
    fl.server.start_server(server_address="localhost:5006", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))


# In[ ]:




