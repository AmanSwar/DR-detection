import sys
sys.path.append("/home/aman/code/research/CV/dia_ret/data_pipeline")

from data_load  import EyepacsGradingDataset , AptosGradingDataset , IdridGradingDataset , DdrGradingDataset


def gradingTester(dataset: str) -> None:

    if dataset == "eyepacs":
        eyepacs = EyepacsGradingDataset()

        eyepacs_train_img , eyepacs_train_labels = eyepacs.get_train_set()
        eyepacs_test_img , eyepacs_test_labels = eyepacs.get_test_set()
        eyepacs_valid_img , eyepacs_valid_labels = eyepacs.get_valid_set()

        print(eyepacs_train_img[:5])
        print(eyepacs_train_labels[:5])

    elif dataset == "aptos":
        aptos = AptosGradingDataset()

        aptos_train_img , aptos_train_labels = aptos.get_train_set()
        aptos_test_img , aptos_test_labels = aptos.get_test_set()
        aptos_valid_img , aptos_valid_labels = aptos.get_valid_set()


        print(aptos_train_img[:5])
        print(aptos_train_labels[:5])

    elif dataset == "ddr":
        ddr = DdrGradingDataset()

        ddr_train_img , ddr_train_labels = ddr.get_train_set()
        ddr_test_img , ddr_test_labels = ddr.get_test_set()
        ddr_valid_img , ddr_valid_labels = ddr.get_valid_set()


        print(ddr_train_img[:5])
        print(ddr_train_labels[:5])

    elif dataset == "idrid":
        idrid = IdridGradingDataset()

        idrid_train_img , idrid_train_labels = idrid.get_train_set()
        idrid_test_img , idrid_test_labels = idrid.get_test_set()
        idrid_valid_img , idrid_valid_labels = idrid.get_valid_set()


        print(idrid_train_img[:5])
        print(idrid_train_labels[:5])


gradingTester("eyepacs")
gradingTester("aptos")
gradingTester("ddr")
gradingTester("idrid")