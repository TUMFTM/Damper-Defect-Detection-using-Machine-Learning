function S = CatStructFields(S, T, dim)
    fields = fieldnames(S);
    for cntFields = 1:numel(fields)
      aField     = fields{cntFields};
      if isstruct(S.(aField))
          afields = fieldnames(S.(aField));
          for cntSubFields = 1:numel(afields)
              bField = afields{cntSubFields};
              S.(aField).(bField) = cat(dim, S.(aField).(bField), T.(aField).(bField));
          end
          continue
      end
      S.(aField) = cat(dim, S.(aField), T.(aField));
    end
end