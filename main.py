import strawberry
import uvicorn
import pickle
import base64
import hashlib
import numpy as np
import puan.ndarray as pnd
import npycvx

from starlette.applications import Starlette
from starlette.config import Config
from strawberry.asgi import GraphQL
from enum import Enum
from typing import List, Union, Optional
from itertools import chain, starmap, repeat

from puan_lite import *

temp_data = {}

@strawberry.enum
class PropositionType(Enum):

    And = "AND"
    Or = "OR"
    Nand = "NAND"
    Nor = "NOR"
    Xor = "XOR"
    Xnor = "XNOR"
    Empt = "EMPTY"

    AtMostOne = "AT_MOST_ONE"
    AllOrNone = "ALL_OR_NONE"

    def to_puan(self):
        return {
            PropositionType.And: And,
            PropositionType.Or: Or,
            PropositionType.Nand: Nand,
            PropositionType.Nor: Nor,
            PropositionType.Xor: Xor,
            PropositionType.Xnor: XNor,
            PropositionType.Empt: Empt,
            PropositionType.AtMostOne: AtMostOne,
            PropositionType.AllOrNone: AllOrNone,
        }[self]

@strawberry.type
class Variable:

    id: str

    def to_puan(self) -> str:
        return self.id

@strawberry.type
class Proposition:
    
    type: PropositionType
    variables: List[Variable]

    def to_puan(self) -> Proposition:
        return self.type.to_puan()(
            *map(
                lambda variable: variable.to_puan(),
                self.variables
            ),
        )

@strawberry.type
class ComplexProposition:

    type: PropositionType
    complex: Optional[List[Proposition]] = None
    variables: Optional[List[Variable]] = None

    def to_puan(self) -> Proposition:

        return self.type.to_puan()(
            *chain(
                map(
                    lambda proposition: proposition.to_puan(),
                    self.complex or []
                ),
                map(
                    lambda variable: variable.to_puan(),
                    self.variables or [],
                )
            ),
        )

@strawberry.type
class Implication:

    condition: Optional[ComplexProposition] = None
    consequence: ComplexProposition

    def to_puan(self) -> Proposition:
        return Impl(
            condition=self.condition.to_puan(),
            consequence=self.consequence.to_puan(),
        ) if self.condition else self.consequence.to_puan()

@strawberry.input
class VariableInput:

    id: str

    def to_output_model(self) -> Variable:
        return Variable(id=self.id)

@strawberry.input
class PropositionInput:
    
    type: PropositionType
    variables: List[VariableInput]

    def to_output_model(self) -> Proposition:
        return Proposition(
            type=self.type,
            variables=list(
                map(
                    lambda variable: Variable(id=variable.id),
                    self.variables
                )
            ),
        )

@strawberry.input
class ComplexPropositionInput:

    type: PropositionType
    complex: Optional[List[PropositionInput]] = None
    variables: Optional[List[VariableInput]] = None

    def to_output_model(self) -> ComplexProposition:
        return ComplexProposition(
            type=self.type,
            complex=list(
                map(
                    lambda proposition: Proposition(
                        type=proposition.type,
                        variables=list(
                            map(
                                lambda variable: Variable(id=variable.id),
                                proposition.variables
                            )
                        )
                    ),
                    self.complex
                )
            ) if self.complex else None,
            variables=list(
                map(
                    lambda variable: Variable(id=variable.id),
                    self.variables
                )
            ) if self.variables else None,
        )

@strawberry.input
class ImplicationInput:

    condition: Optional[ComplexPropositionInput] = None
    consequence: ComplexPropositionInput

    def to_output_model(self) -> Implication:
        return Implication(
            condition=self.condition.to_output_model(),
            consequence=self.consequence.to_output_model(),
        )


@strawberry.type
class ValuedVariable:

    id: str
    value: int

@strawberry.type
class Interpretation:

    variables: List[ValuedVariable]

    @staticmethod
    def from_dict(interpretation: dict) -> "Interpretation":
        return Interpretation(
            variables=list(
                starmap(
                    lambda id, val: ValuedVariable(
                        id=id,
                        value=val,
                    ),
                    interpretation.items(),
                )
            )
        )

@strawberry.input
class ValuedVariableInput:

    id: str
    value: int

@strawberry.input
class InterpretationInput:

    variables: List[ValuedVariableInput]

    def to_dict(self) -> dict:
        return dict(
            map(
                lambda variable: (variable.id, variable.value),
                self.variables
            )
        )

@strawberry.enum
class Direction(Enum):
    
    Negative = "NEGATIVE"
    Positive = "POSITIVE"

@strawberry.input
class PreConfigurationInput:

    proposition: ComplexPropositionInput
    valued_variables: List[ValuedVariableInput]

    def to_output_model(self) -> "PreConfiguration":
        return PreConfiguration(
            proposition=self.proposition.to_output_model(),
            valued_variables=list(
                map(
                    lambda variable: ValuedVariable(
                        id=variable.id,
                        value=variable.value,
                    ),
                    self.valued_variables
                )
            )
        )

@strawberry.input
class ConfiguratorSettingsInput:

    default_direction: Direction
    preconfigurations: List[PreConfigurationInput]

    def configuration_settings(self) -> "ConfiguratorSettings":
        return ConfiguratorSettings(
            default_direction=self.default_direction,
            preconfigurations=list(
                map(
                    lambda preconfiguration: preconfiguration.to_output_model(),
                    self.preconfigurations
                )
            )
        )

@strawberry.type
class PreConfiguration:

    proposition: ComplexProposition
    valued_variables: List[ValuedVariable]

    def interpretation_dict(self, interpretation: dict) -> dict:
        polyhedron = And(self.proposition.to_puan()).to_ge_polyhedron()
        boolean_vector = (polyhedron.A.construct(interpretation) >= 1)
        if (polyhedron.A.dot(boolean_vector) >= polyhedron.b).all():
            return dict(
                map(
                    lambda variable: (variable.id, variable.value),
                    self.valued_variables,
                )
            )
        return {}

@strawberry.type
class ConfiguratorSettings:

    default_direction: Direction
    preconfigurations: List[PreConfiguration]

    def interpretation_dicts(self, variables: list, interpretation: dict) -> List[Dict[str, int]]:
        return list(
            chain(
                [
                    dict(
                        zip(
                            map(
                                lambda variable: variable.id,
                                variables,
                            ),
                            repeat(-1 if self.default_direction == Direction.Negative else 1),
                        )
                    )
                ],
                map(
                    lambda pc: pc.interpretation_dict(
                        interpretation,
                    ),
                    self.preconfigurations,
                )
            )
        )

@strawberry.type
class Configurator:

    polyhedron: strawberry.Private[pnd.ge_polyhedron]
    settings: ConfiguratorSettings

    @strawberry.field
    def variables(self) -> List[Variable]:
        return sorted(
            map(
                lambda vr: Variable(id=vr.id),
                self.polyhedron.A.variables,
            ),
            key=lambda variable: variable.id,
        )

    @strawberry.field
    def select(self, prioritization: InterpretationInput) -> List[Interpretation]:
        objective = pnd.integer_ndarray(
            np.vstack(
                list(
                    map(
                        self.polyhedron.A.construct,
                        chain(
                            self.settings.interpretation_dicts(
                                self.polyhedron.A.variables,
                                prioritization.to_dict(),
                            ),
                            [
                                prioritization.to_dict(),
                            ],
                        )
                    )
                )
            )
        ).ndint_compress(method="shadow", axis=0)

        solve_part_fn = functools.partial(
            npycvx.solve_lp, 
            *npycvx.convert_numpy(
                self.polyhedron.A, 
                self.polyhedron.b,
            ), 
            False,
        )

        return list(
            map(
                lambda x: Interpretation.from_dict(
                    dict(
                        zip(
                            map(
                                lambda vr: vr.id,
                                self.polyhedron.A.variables,
                            ),
                            x[1],
                        )
                    )
                ),
                map(
                    solve_part_fn, 
                    [objective],
                )
            )
        )


@strawberry.type
class PropositionStrawberry:

    propositions: List[Implication]

    def to_puan(self) -> And:
        return And(
            *map(
                lambda implication: implication.to_puan(),
                self.propositions,
            )
        )

    @strawberry.field
    def solve(self, interpretations: List[InterpretationInput]) -> List[Interpretation]:
        return list(
            map(
                Interpretation.from_dict,
                self.to_puan().solve(
                    *map(
                        lambda inter: inter.to_dict(),
                        interpretations
                    )
                )
            )
        )

    @strawberry.field
    def configure(self, settings: ConfiguratorSettingsInput) -> Configurator:
        return Configurator(
            polyhedron=self.to_puan().to_ge_polyhedron(),
            settings=settings.configuration_settings(),
        )

    @strawberry.field
    def configureFromId(self, id: str) -> Configurator:
        return Configurator(
            polyhedron=self.to_puan().to_ge_polyhedron(),
            settings=temp_data[id],
        )

    @strawberry.field
    def size(self) -> int:
        return self.to_puan().to_ge_polyhedron().size

    @strawberry.field
    def shape(self) -> List[int]:
        return list(self.to_puan().to_ge_polyhedron().shape)

    @strawberry.field
    def variables(self) -> List[Variable]:
        return sorted(
            map(
                lambda s: Variable(id=s),
                self.to_puan().atoms(),
            ),
            key=lambda variable: variable.id,
        )

@strawberry.type
class Query:
    
    @strawberry.field
    def model_from_propositions(self, propositions: List[ImplicationInput]) -> PropositionStrawberry:
        return PropositionStrawberry(
            propositions=list(
                map(
                    lambda implication: Implication(
                        condition=implication.condition.to_output_model() if implication.condition else None,
                        consequence=implication.consequence.to_output_model(),
                    ),
                    propositions
                )
            ),
        )

    @strawberry.field
    def model_from_id(self, id: str) -> Optional[PropositionStrawberry]:
        return temp_data.get(id, None)

@strawberry.type
class Mutation:

    @strawberry.mutation
    def add_model(self, propositions: List[ImplicationInput]) -> str:
        model = PropositionStrawberry(
            propositions=list(
                map(
                    lambda implication: Implication(
                        condition=implication.condition.to_output_model() if implication.condition else None,
                        consequence=implication.consequence.to_output_model(),
                    ),
                    propositions
                )
            ),
        )
        id = hashlib.sha256(pickle.dumps(model.to_puan())).hexdigest()
        temp_data[id] = model
        return id

    @strawberry.mutation
    def add_configuration_setting(self, settings: ConfiguratorSettingsInput) -> str:
        id = hashlib.sha256(pickle.dumps(settings.configuration_settings())).hexdigest()
        temp_data[id] = settings.configuration_settings()
        return id

config = Config(".env")

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
)
graphql_app = GraphQL(schema)

app = Starlette(debug=config('DEBUG', cast=bool, default=False))
app.add_route("/graphql", graphql_app)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=config("API_PORT", cast=int, default=8000))