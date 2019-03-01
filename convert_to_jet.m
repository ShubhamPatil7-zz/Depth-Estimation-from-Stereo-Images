function convert_to_jet()
    file_list = dir('./*.png');
    for counter = 1 : length( file_list )
        fn = file_list(counter).name;
        img = im2double(imread( fn ));
        figure; imagesc(img); colormap jet; axis image; axis off;
        % pause(0.00000001);
        saveas(gcf, [ './colored/' fn ]); 
    end   
end