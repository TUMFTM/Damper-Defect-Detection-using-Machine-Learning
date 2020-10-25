function [] = setPowerMode(mode)
    if ~any(strcmp(mode, {'balance','save','power'}))
        error('Unknown power mode.')
    end
    powercfg.balance = 'SCHEME_BALANCED';
    powercfg.save    = 'SCHEME_MAX';
    powercfg.power   = 'SCHEME_MIN';
    [status,text] = system(['powercfg -setactive ', powercfg.(mode)]);
    if status == 0
        [status,text]=system(['powercfg /L ']);
        endMode = strfind(text, '*'); 
        endMode = endMode(end);
        startMode = strfind(text, '(');
        startMode = startMode(find(startMode < endMode, 1, 'last'));
        disp(sprintf('Active power mode: "%s"',text(startMode+1:endMode-3)))
    end    
end