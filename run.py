import torch
from pathlib import Path

from cyp.models import ConvModel, RNNModel

# import fire


class RunTask:
    """Entry point into the pipeline.

    For convenience, all the parameter descriptions are copied from the classes.
    """

    @staticmethod
    def train_cnn(
        datapath,
        dropout=0.5,
        dense_features=None,
        savedir=Path("data/models"),
        num_runs=2,
        train_steps=25000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=1,
        l1_weight=0,
        patience=10,
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Train a CNN model

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to which histogram has been saved
        dropout: float, default=0.5
            Default taken from the original paper
        dense_features: list, or None, default=None.
            output feature size of the Linear layers. If None, default values will be taken from the paper.
            The length of the list defines how many linear layers are used.
        savedir: pathlib Path, default=Path('data/models')
            The directory into which the models should be saved.
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int or None, default=None
            Which year to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=25000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            In addition to MSE, L1 loss is also used (sometimes). The default is 0, but a value of 1.5 is used
            when training the model in batch
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.

        use_gp: boolean, default=True
            Whether to use a Gaussian process in addition to the model

        If use_gp=True, the following parameters are also used:

        sigma: float, default=1
            The kernel variance, or the signal variance
        r_loc: float, default=0.5
            The length scale for the location data (latitudes and longitudes)
        r_year: float, default=1.5
            The length scale for the time data (years)
        sigma_e: float, default=0.32
            Noise variance. 0.32 **2 ~= 0.1
        sigma_b: float, default=0.01
            Parameter variance; the variance on B

        device: torch.device
            Device to run model on. By default, checks for a GPU. If none exists, uses
            the CPU

        """

        model = ConvModel(
            in_channels=9,
            dropout=dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
        )
        model.run(
            datapath,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

    @staticmethod
    def train_rnn(
        datapath="../geomaml/yield/data",
        num_bins=32,
        hidden_size=128,
        rnn_dropout=0.75,
        dense_features=None,
        savedir=Path("data/models"),
        num_runs=2,
        train_steps=10000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Train an RNN model

        Parameters
        ----------
        cleaned_data_path: str, default='data/img_output'
            Path to which histogram has been saved
        num_bins: int, default=32
            Number of bins in the generated histogram
        hidden_size: int, default=128
            The size of the hidden state. Default taken from the original paper
        rnn_dropout: float, default=0.75
            Default taken from the original paper. Note that this dropout is applied to the
            hidden state after each timestep, not after each layer (since there is only one layer)
        dense_features: list, or None, default=None.
            output feature size of the Linear layers. If None, default values will be taken from the paper.
            The length of the list defines how many linear layers are used.
        savedir: pathlib Path, default=Path('data/models')
            The directory into which the models should be saved.
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=10000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            L1 loss is not used for the RNN. Setting it to 0 avoids it being computed.
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.

        use_gp: boolean, default=True
            Whether to use a Gaussian process in addition to the model

        If use_gp=True, the following parameters are also used:

        sigma: float, default=1
            The kernel variance, or the signal variance
        r_loc: float, default=0.5
            The length scale for the location data (latitudes and longitudes)
        r_year: float, default=1.5
            The length scale for the time data (years)
        sigma_e: float, default=0.32
            Noise variance. 0.32 **2 ~= 0.1
        sigma_b: float, default=0.01
            Parameter variance; the variance on B

        device: torch.device
            Device to run model on. By default, checks for a GPU. If none exists, uses
            the CPU

        """
        model = RNNModel(
            in_channels=9,
            num_bins=num_bins,
            hidden_size=hidden_size,
            rnn_dropout=rnn_dropout,
            dense_features=dense_features,
            savedir=savedir,
            use_gp=use_gp,
            sigma=sigma,
            r_loc=r_loc,
            r_year=r_year,
            sigma_e=sigma_e,
            sigma_b=sigma_b,
            device=device,
        )
        model.run(
            datapath,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )


if __name__ == "__main__":
    # fire.Fire(RunTask)
    RunTask.train_rnn()
