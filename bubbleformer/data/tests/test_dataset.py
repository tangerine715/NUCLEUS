import pytest
from bubbleformer.data import BubblemlForecast

@pytest.mark.parametrize("fields", [
    ["dfun"],
    ["temperature"],
    ["dfun", "temperature"],
    ["dfun", "temperature", "velx"],
    ["dfun", "velx", "vely"],
    ["dfun", "temperature", "velx", "vely"]
])
@pytest.mark.parametrize("norm", ["none", "std", "minmax", "tanh"])
@pytest.mark.parametrize("time_window", [5, 10])
def test_bubblemlforecastdataset(fields, norm, time_window):
    """
    Test the BubblemlForecast dataset
    The samples are 2 50x64x64 (TxHxW) trajectories 
    """
    dataset = BubblemlForecast(
        filenames=["samples/sample_1.hdf5", "samples/sample_2.hdf5"],
        fields=fields,
        norm=norm,
        time_window=time_window,
        start_time=5
    )
    diff_term, div_term = dataset.normalize()
    sample = dataset[0]

    expected_shape = (time_window, len(fields), 64, 64)

    assert len(dataset) == 2 * (50 - 5 - 2 * time_window + 1)
    assert sample[0].shape == expected_shape and sample[1].shape == expected_shape
    assert diff_term.shape == (len(fields),) and div_term.shape == (len(fields),)
