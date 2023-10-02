# Puan Lite API

## Example
Create a model
```gql
mutation CreateModel{
   addModel(propositions: [
      {
        condition: {
          type: Or,
          variables: [
            {id: "a"},
            {id: "b"}
          ]
        },
        consequence: {
          type: Xor, 
          complex: [
            {
              type: And,
              variables: [
                {id: "p"}
                {id: "q"}
              ]
            }
            {
              type: And,
              variables: [
                {id: "r"}
                {id: "s"}
              ]
            }
          ]
        }
      }
    ])
}
```
Copy the key hash returned from the query. Start using configurator backend
```gql
query Configurator{
  modelFromId(id: "1bda48cd5b810298555514b7beecdb52bff8808d5ec7d5ea92310aed9ffdd78b") {
    configure(settings: {
      defaultDirection: Negative
    }) {
      select(
        prioritation: {
          variables: [
            {
              id: "a",
              value: 1
            },
            {
              id: "r",
              value: 2
            }
          ]
        }
      ) {
        variables {
          id
          value
        }
      }
    }
  }
}
```
