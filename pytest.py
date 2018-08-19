def generator(start, total_samples, which_gen):
    main_index = start
    documents = read_csv('document.csv')

    while True:
        batch_features = np.zeros((batch_size, Image_width, Image_height, 3))
        batch_labels = np.zeros((batch_size, num_classes))

        for i in range(batch_size):
            row = documents[main_index]
            document_id = row[0]
            document_type = map_document_type(row[1],unmapped_document_types[unmapped_document_types_ids.index(row[1])][1])
            document_dir = f"D:\\documents\\{document_type}"
            my_dir = Path(document_dir)
            if not my_dir.is_dir():
                my_dir.mkdir()
            document_path = f"{document_dir}\\{document_id}.jpg"
            my_file = Path(document_path)
            if not my_file.is_file():
                print(document_url)
                document_url = f"https://apis.emaratech.ae/v1/UChannel/services/document/{document_id}"
                urllib.request.urlretrieve(document_url, document_path)
                #time.sleep(2)
            document_conv_path = f"{document_dir}\\{document_id}_{Image_width}_{Image_height}.jpg"
            my_conv_file = Path(document_conv_path)
            if not my_conv_file.is_file():
                img = Image.open(document_path).resize((Image_width, Image_height), Image.ANTIALIAS)
                img.save(document_conv_path, quality=100)

            img = Image.open(document_conv_path)
            #arr = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
            arr = np.asarray(img)

            batch_features[i] = arr
            batch_labels[i] = [1 if j == document_types.index(document_type) else 0 for j in range(num_classes)]
            main_index = main_index + 1
            print(which_gen + ' ' + str(main_index))
            if main_index-start == total_samples:
                main_index = start

        yield (batch_features, batch_labels)


unmapped_document_types = read_csv('document_type.csv')
unmapped_document_types_ids = [u[0] for u in unmapped_document_types]
document_types = [o for o in set([ map_document_type(t[0],t[1]) for t in unmapped_document_types])]
