
import traceback 


from data_pipeline.data_load import EyepacsGradingDataset , AptosGradingDataset , IdridGradingDataset , DdrGradingDataset


def gradingTester(dataset: str) -> None:

    if dataset == "eyepacs":
        try:
            eyepacs = EyepacsGradingDataset()

            eyepacs_train_img , eyepacs_train_labels = eyepacs.get_train_set()
            eyepacs_test_img , eyepacs_test_labels = eyepacs.get_test_set()
            eyepacs_valid_img , eyepacs_valid_labels = eyepacs.get_valid_set()

            print(eyepacs_train_img[:5])
            # print(eyepacs_train_labels[:5])

            all_img = [img for sublist in eyepacs_test_img for img in sublist]

            print(all_img[:5])
        except Exception as e:
            print("Error in eyepacs")


    elif dataset == "aptos":
        try:
            aptos = AptosGradingDataset()

            aptos_train_img , aptos_train_labels = aptos.get_train_set()
            aptos_test_img , aptos_test_labels = aptos.get_test_set()
            aptos_valid_img , aptos_valid_labels = aptos.get_valid_set()


            print(aptos_train_img[:5])
            print(aptos_train_labels[:5])

        except Exception as e:
            print("Error in aptos")

    elif dataset == "ddr":
        try:
            ddr = DdrGradingDataset(root_dir="data/ddr")

            ddr_train_img , ddr_train_labels = ddr.get_train_set()
            ddr_test_img , ddr_test_labels = ddr.get_test_set()
            ddr_valid_img , ddr_valid_labels = ddr.get_valid_set()


            print(ddr_train_img[:5])
            print(ddr_train_labels[:5])

        except Exception as e:
            print("error in ddr") 
            print(e)
            traceback.print_exc()

    elif dataset == "idrid":
        try:        
            idrid = IdridGradingDataset()

            idrid_train_img , idrid_train_labels = idrid.get_train_set()
            idrid_test_img , idrid_test_labels = idrid.get_test_set()
            idrid_valid_img , idrid_valid_labels = idrid.get_valid_set()


            print(idrid_train_img[:5])
            print(idrid_train_labels[:5])


        except Exception as e:
            print("error in idrid dataset")
            # traceback.print_exc()

print("\nPrinting eyepac ..")
gradingTester("eyepacs")
print("\nprinting aptos...")
gradingTester("aptos")
print("\nprinting ddr")
gradingTester("ddr")
print("\nprinting")
gradingTester("idrid")      



