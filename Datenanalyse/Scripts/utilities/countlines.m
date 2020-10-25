% COUNTLINES count lines of code in .m, files. COUNTLINES('FILENAME')
% counts the total number of lines of code in the file specified by
% FILENAME. It does not count blank lines or comments. 
%
% COUNTLINES('ALL') counts the total number of lines of code in all of the
% .m files in the current directory. Modify the regular expression 
% for including other file extensions, if necessary.
    
    function count = countlines(filename)
    count = 0; 

    if (strcmp(filename,'all'))
        %files = dir(cd);
        files = dir('**/*');

        for lv = 1:length(files)
            if (isempty(regexp(files(lv).name,'\w*\.[m]', 'once')))
                continue
            end
            temp = countlines(fullfile(files(lv).folder, files(lv).name));
            count = temp+count;

        end
    else
        fid = fopen(filename,'r');

        while(~feof(fid))
            line_read = fgetl(fid);
            if ~(iscomment(line_read)) && ~iswhitespace(line_read)
                count = count+1;
            end
            
        end
        fclose(fid);
    end
    end
    
    % subfunction to determine if a line is a comment line only
    % throws away leading white space using regexp
    function yesno = iscomment(str)
        yesno = ~isempty(regexp(str,'^\s*%'));
    
    end
    
    function yesno = iswhitespace(str)
        yesno = isempty(regexp(str,'\w*'));
    
    end

