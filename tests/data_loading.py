
import traceback 


from data_pipeline.data_load import EyepacsGradingDataset , AptosGradingDataset , IdridGradingDataset , DdrGradingDataset, MessdrGradingDataset


def gradingTester(dataset: str) -> None:

    if dataset == "eyepacs":
        try:
            eyepacs = EyepacsGradingDataset()

            eyepacs_train_img , eyepacs_train_labels = eyepacs.get_train_set()
            eyepacs_test_img , eyepacs_test_labels = eyepacs.get_test_set()
            eyepacs_valid_img , eyepacs_valid_labels = eyepacs.get_valid_set()

            print(eyepacs_train_img[:5])
            print(eyepacs_train_labels[:5])
            
            print("\n")

            print(f"length of train , valid , test {dataset}")
            print(len(eyepacs_train_img))
            print(len(eyepacs_valid_img))
            print(len(eyepacs_test_img))




            # all_img = [img for sublist in eyepacs_test_img for img in sublist]

            # print(all_img[:5])
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

            print("\n")

            print(f"length of train , valid , test {dataset}")
            print(len(aptos_train_img))
            print(len(aptos_valid_img))
            print(len(aptos_test_img))


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

            print("\n")

            print(f"length of train , valid , test {dataset}")
            print(len(ddr_train_img))
            print(len(ddr_valid_img))
            print(len(ddr_test_img))


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

            print("\n")

            print(f"length of train , valid , test {dataset}")
            print(len(idrid_train_img))
            print(len(idrid_valid_img))
            print(len(idrid_test_img))


        except Exception as e:
            print("error in idrid dataset")
            # traceback.print_exc()

    elif dataset == "messdr":
        try:
            messdr = MessdrGradingDataset()
            messdr_train_img , messdr_train_labels = messdr.get_train_set()
            messdr_test_img , messdr_test_labels = messdr.get_test_set()
            messdr_valid_img , messdr_valid_labels = messdr.get_valid_set()
            print(messdr_train_img[:5])
            print(messdr_train_labels[:5])

            print("\n")

            print(f"length of train , valid , test {dataset}")
            print(len(messdr_train_img))
            print(len(messdr_valid_img))
            print(len(messdr_test_img))

            
        except Exception as e:
            print("error in messdr")
            traceback.print_exc()

print("\nPrinting eyepac ..")
gradingTester("eyepacs")
print("\nprinting aptos...")
gradingTester("aptos")
print("\nprinting ddr")
gradingTester("ddr")
print("\nprinting")
gradingTester("idrid")      

print("\nprining messidiro")
gradingTester("messdr")



